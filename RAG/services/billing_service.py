"""
Billing / Plans / Usage Limits

Internal bookkeeping only - no real payment processor, no card
charging. A Super Admin manually assigns a Plan to an Organization or a
Personal Workspace User (billing_views.py/personal_billing_views.py);
this module enforces the assigned Plan's limits with a hard block once
exceeded, and reports current usage. An Owner (or Personal user) can
also REQUEST a plan change themselves (request_plan_change()), which
stays Pending until a Platform Admin approves it (approve_plan_request())
- direct assignment by a Platform Admin (assign_plan()) is a separate,
instant, admin-initiated path that skips the request queue entirely.

No Subscription row for an organization/user means unlimited - the
caller (get_active_subscription/get_active_personal_subscription) never
has to special-case "new organization" vs. "explicitly unlimited
organization", and every check_*_limit below returns immediately (no
query, no block) when that's the case. This is the same "absence means
today's behavior, nothing breaks until someone opts in" rollout
convention this codebase uses elsewhere for additive, nullable rollout
fields. The instant ANY Plan is assigned, though, AI credits become a
strictly metered resource governed by that Plan's `included_credits`
(refilled every billing period - see _advance_period()) - credits are
no longer an independent axis an Owner can leave untouched forever once
a Plan is in play, unlike before this module supported Plan-integrated
credits.

Usage is computed by aggregating QueryLog/AITaskRun/Document/
OrganizationMembership directly - NOT a separately-maintained counter
row. Those tables are already the authoritative record of what
happened (QueryLog IS the Search History/Analytics source of truth,
AITaskRun IS the AI Task History source of truth); a second counter
that increments alongside them is a second source of truth that can
drift from the first (e.g. if a write to one succeeds and the other
doesn't), exactly the class of bug this codebase's other authorization
primitives (get_accessible_document_ids, ORG_PERMISSION_MIN_ROLE) are
built to avoid. The trade-off: get_organization_usage() is cached for
a short, fixed window (RETRIEVAL_CACHE-style precedent -
context_processors.sidebar_status() caches 30s, system_config_service
caches 15s) rather than computed fresh on every single request, so
there's a small, bounded window where slightly-over-limit usage can
slip through before a block takes effect. Accepted given these are not
per-second-frequency actions (asking a question, starting an AI Task
run, uploading a document, inviting a member).

Company (`organization=`) and Personal (`user=`) are two entirely
parallel, never-mixed systems throughout this module - a Company's
credit pool/usage/Plan and a user's own Personal Workspace credit
pool/usage/Plan share nothing, by design (requirement: "Personal
Workspace credits and Company Workspace credits must remain completely
separate"). Rather than risk changing the signature/behavior of the
original, heavily-tested `organization=`-scoped functions, each has a
`_personal`-suffixed sibling with identical logic keyed by `user`/
`UserProfile.ai_credits_balance` instead of `organization`/
`Organization.ai_credits_balance` - both delegate to the same private
`_for()`-style core so the two families can't silently drift apart.
"""

import calendar
import logging

from django.core.cache import cache
from django.db.models import Sum
from django.utils import timezone

from ..models import FEATURE_CODES, AICreditTransaction, AITaskRun, Document, Organization, OrganizationMembership, Plan, PlanChangeRequest, QueryLog, Subscription, UserProfile
from .activity_log_service import log_activity
from .org_audit_log_service import log_org_activity

logger = logging.getLogger(__name__)

USAGE_CACHE_SECONDS = 20


class BillingServiceError(Exception):
    pass


class UsageLimitExceeded(Exception):
    def __init__(self, limit_type, message):
        self.limit_type = limit_type
        super().__init__(message)


def _add_one_month(dt):
    """Whole-month step using stdlib `calendar` (no new dependency) - Jan 31 + 1 month lands on Feb 28/29, calendar.monthrange's own documented behavior for a short month, not a bug to special-case."""
    year = dt.year + (dt.month // 12)
    month = dt.month % 12 + 1
    day = min(dt.day, calendar.monthrange(year, month)[1])
    return dt.replace(year=year, month=month, day=day)


def _add_one_year(dt):
    """Whole-year step, same clamping approach as _add_one_month (Feb 29 -> Feb 28 on a non-leap target year)."""
    year = dt.year + 1
    day = min(dt.day, calendar.monthrange(year, dt.month)[1])
    return dt.replace(year=year, day=day)


def _advance_period_end(period_end, plan):
    """Next period boundary, respecting the Plan's billing_interval (monthly/yearly) - the one place period length is decided, so a Plan's interval change only ever affects FUTURE rollovers, never rewrites past periods."""
    if plan.billing_interval == Plan.BillingInterval.YEARLY:
        return _add_one_year(period_end)
    return _add_one_month(period_end)


# ============================================================
# Subscription lookup - one private core, two public scopes
# ============================================================

def _active_subscription_for(organization=None, user=None):
    qs = Subscription.objects.select_related("plan")
    if organization is not None:
        return qs.filter(organization=organization).first()
    if user is not None:
        return qs.filter(user=user).first()
    return None


def get_active_subscription(organization):
    """None for Personal Workspace (organization=None) or an org with no Subscription row - both mean unlimited. Unchanged signature/behavior from before Personal Plans existed."""
    if organization is None:
        return None
    return _active_subscription_for(organization=organization)


def get_active_personal_subscription(user):
    """The Personal-Workspace counterpart to get_active_subscription() - None means unlimited, same contract."""
    if user is None:
        return None
    return _active_subscription_for(user=user)


def _credit_holder(subscription):
    """Returns the object (Organization or UserProfile) whose ai_credits_balance this subscription's credits live on."""
    if subscription.organization_id:
        return subscription.organization
    return subscription.user.profile


def _refill_credits_for_subscription(subscription, reason="Plan period refill"):
    """
    Resets the subscription's credit balance to EXACTLY plan.
    included_credits (not additive) - a deliberate "top up TO the
    plan's amount" reset, not a cumulative grant, so unused credits
    (included or extra-purchased) don't silently compound period over
    period. Writes one AICreditTransaction capturing whatever the net
    delta was (positive if this raises the balance, negative if it
    lowers it - e.g. unused extra-purchased credits from the prior
    period being cleared). Called on genuine period rollover
    (_advance_period) and on first-ever Plan assignment (assign_plan/
    the personal equivalent) alike, so a brand new Subscription starts
    funded immediately rather than waiting for the next boundary.
    """

    from django.db import transaction

    included = subscription.plan.included_credits
    with transaction.atomic():
        if subscription.organization_id:
            locked = Organization.objects.select_for_update().get(pk=subscription.organization_id)
            previous = locked.ai_credits_balance or 0
            locked.ai_credits_balance = included
            locked.save(update_fields=["ai_credits_balance"])
            delta = included - previous
            if delta != 0:
                AICreditTransaction.objects.create(
                    organization=locked, amount=delta, balance_after=included, reason=reason,
                )
        else:
            profile = UserProfile.objects.select_for_update().get(user_id=subscription.user_id)
            previous = profile.ai_credits_balance or 0
            profile.ai_credits_balance = included
            profile.save(update_fields=["ai_credits_balance"])
            delta = included - previous
            if delta != 0:
                AICreditTransaction.objects.create(
                    user=subscription.user, amount=delta, balance_after=included, reason=reason,
                )


def _advance_period(subscription):
    """
    Lazily rolls current_period_start/end forward to contain `now`,
    stepping by the Plan's billing_interval (monthly/yearly - see
    _advance_period_end()). This IS the "reset at period boundaries"
    mechanism - there is no scheduled job anywhere (this app has no
    Celery/cron beyond task_runner.py's in-process thread pool, which
    this doesn't touch): the next time ANY check reads this
    subscription after its period has lapsed, the window silently
    advances, usage recomputes against the new window directly from
    QueryLog/AITaskRun (nothing to "reset" because nothing is stored
    as a running counter), AND the credit balance refills to the
    Plan's included_credits (see _refill_credits_for_subscription()) -
    exactly once per boundary crossed, even if multiple periods have
    lapsed since the last read (the while loop below), so credits
    never compound from being read infrequently.
    """
    now = timezone.now()
    changed = False
    while now >= subscription.current_period_end:
        subscription.current_period_start = subscription.current_period_end
        subscription.current_period_end = _advance_period_end(subscription.current_period_end, subscription.plan)
        changed = True
    if changed:
        subscription.save(update_fields=["current_period_start", "current_period_end"])
        _refill_credits_for_subscription(subscription)
    return subscription.current_period_start, subscription.current_period_end


def _serialize_plan_snapshot(plan):
    return {
        "id": plan.id,
        "name": plan.name,
        "slug": plan.slug,
        "plan_type": plan.plan_type,
        "price": str(plan.price) if plan.price is not None else None,
        "currency": plan.currency,
        "billing_interval": plan.billing_interval,
        "max_queries_per_month": plan.max_queries_per_month,
        "max_ai_task_runs_per_month": plan.max_ai_task_runs_per_month,
        "max_storage_bytes": plan.max_storage_bytes,
        "max_seats": plan.max_seats,
        "included_credits": plan.included_credits,
        "allow_credit_purchase": plan.allow_credit_purchase,
        "included_features": plan.included_features,
    }


def get_organization_usage(organization):
    subscription = get_active_subscription(organization)
    if subscription is None or not subscription.plan.is_active:
        return {"plan": None, "unlimited": True, "credits_balance": organization.ai_credits_balance if organization else None}

    period_start, period_end = _advance_period(subscription)
    cache_key = f"org_usage:{organization.id}:{period_start.isoformat()}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    plan = subscription.plan
    organization.refresh_from_db(fields=["ai_credits_balance"])
    usage = {
        "unlimited": False,
        "plan": _serialize_plan_snapshot(plan),
        "period_start": period_start,
        "period_end": period_end,
        "queries_used": QueryLog.objects.filter(
            organization=organization, created_at__gte=period_start, created_at__lt=period_end,
        ).count(),
        "ai_task_runs_used": AITaskRun.objects.filter(
            organization=organization, created_at__gte=period_start, created_at__lt=period_end,
        ).count(),
        "storage_used_bytes": Document.objects.filter(organization=organization).aggregate(t=Sum("file_size"))["t"] or 0,
        "seats_used": OrganizationMembership.objects.filter(
            organization=organization, status=OrganizationMembership.Status.ACTIVE,
        ).count(),
        "credits_balance": organization.ai_credits_balance,
    }
    cache.set(cache_key, usage, USAGE_CACHE_SECONDS)
    return usage


def get_personal_usage(user):
    """The Personal-Workspace counterpart to get_organization_usage() - same shape, minus seats_used (a Personal Workspace has no members)."""
    subscription = get_active_personal_subscription(user)
    profile = user.profile
    if subscription is None or not subscription.plan.is_active:
        return {"plan": None, "unlimited": True, "credits_balance": profile.ai_credits_balance}

    period_start, period_end = _advance_period(subscription)
    cache_key = f"personal_usage:{user.id}:{period_start.isoformat()}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    plan = subscription.plan
    profile.refresh_from_db(fields=["ai_credits_balance"])
    usage = {
        "unlimited": False,
        "plan": _serialize_plan_snapshot(plan),
        "period_start": period_start,
        "period_end": period_end,
        "queries_used": QueryLog.objects.filter(
            user=user, organization__isnull=True, created_at__gte=period_start, created_at__lt=period_end,
        ).count(),
        "ai_task_runs_used": AITaskRun.objects.filter(
            user=user, organization__isnull=True, created_at__gte=period_start, created_at__lt=period_end,
        ).count(),
        "storage_used_bytes": Document.objects.filter(user=user, organization__isnull=True).aggregate(t=Sum("file_size"))["t"] or 0,
        "credits_balance": profile.ai_credits_balance,
    }
    cache.set(cache_key, usage, USAGE_CACHE_SECONDS)
    return usage


def _check(organization, used_key, limit_key, limit_type, message_noun):
    if organization is None:
        return
    usage = get_organization_usage(organization)
    if usage["unlimited"]:
        return
    limit = usage["plan"][limit_key]
    if limit is not None and usage[used_key] >= limit:
        raise UsageLimitExceeded(
            limit_type,
            f"You've reached this organization's {message_noun} limit ({limit}). "
            f"Upgrade your plan to continue.",
        )


def check_query_limit(organization):
    _check(organization, "queries_used", "max_queries_per_month", "queries", "monthly AI query")


def check_ai_task_limit(organization):
    _check(organization, "ai_task_runs_used", "max_ai_task_runs_per_month", "ai_task_runs", "monthly AI Task run")


def check_storage_limit(organization, additional_bytes):
    if organization is None:
        return
    usage = get_organization_usage(organization)
    if usage["unlimited"]:
        return
    limit = usage["plan"]["max_storage_bytes"]
    if limit is not None and usage["storage_used_bytes"] + additional_bytes > limit:
        raise UsageLimitExceeded(
            "storage",
            f"This upload would exceed this organization's storage limit ({limit} bytes). Upgrade your plan to continue.",
        )


def check_seat_limit(organization, additional_seats=1):
    if organization is None:
        return
    usage = get_organization_usage(organization)
    if usage["unlimited"]:
        return
    limit = usage["plan"]["max_seats"]
    if limit is not None and usage["seats_used"] + additional_seats > limit:
        raise UsageLimitExceeded(
            "seats",
            f"This organization has reached its seat limit ({limit}). Upgrade your plan to invite more members.",
        )


# ---- Personal-Workspace counterparts of the four checks above ----

def _check_personal(user, used_key, limit_key, limit_type, message_noun):
    if user is None:
        return
    usage = get_personal_usage(user)
    if usage["unlimited"]:
        return
    limit = usage["plan"][limit_key]
    if limit is not None and usage[used_key] >= limit:
        raise UsageLimitExceeded(
            limit_type,
            f"You've reached your personal workspace's {message_noun} limit ({limit}). "
            f"Upgrade your plan to continue.",
        )


def check_personal_query_limit(user):
    _check_personal(user, "queries_used", "max_queries_per_month", "queries", "monthly AI query")


def check_personal_ai_task_limit(user):
    _check_personal(user, "ai_task_runs_used", "max_ai_task_runs_per_month", "ai_task_runs", "monthly AI Task run")


def check_personal_storage_limit(user, additional_bytes):
    if user is None:
        return
    usage = get_personal_usage(user)
    if usage["unlimited"]:
        return
    limit = usage["plan"]["max_storage_bytes"]
    if limit is not None and usage["storage_used_bytes"] + additional_bytes > limit:
        raise UsageLimitExceeded(
            "storage",
            f"This upload would exceed your personal workspace's storage limit ({limit} bytes). Upgrade your plan to continue.",
        )


# ============================================================
# AI Credits - a spendable balance now GRANTED BY (and refilled every
# billing period from) the org's/user's active Plan, rather than an
# independent axis - see this module's docstring. An organization/user
# with no active Subscription is still fully unmetered (NULL balance =
# unlimited), matching the Plan-based checks' own "no row = unlimited"
# contract now, unlike the pre-redesign version of this module where
# credits stayed metered even with no Plan at all.
# ============================================================

AI_CREDIT_COST_PER_QUERY = 1
# AI Tasks span a variable number of documents, so a flat cost would
# undercharge a 50-document run relative to a 1-document one - one
# credit per document processed, floored at 1 so even a 0-document
# edge case still costs something.
AI_CREDIT_COST_PER_AI_TASK_DOCUMENT = 1


def check_ai_credits(organization, cost):
    """Organization=None (Personal Workspace) is never subject to COMPANY credits - this is an organization-level concept only, same carve-out every check_*_limit above already makes. A NULL balance (no Plan ever assigned, or no Owner has topped up) means unlimited - see Organization.ai_credits_balance's help_text."""

    if organization is None or organization.ai_credits_balance is None:
        return
    if organization.ai_credits_balance < cost:
        raise UsageLimitExceeded(
            "ai_credits",
            f"This organization doesn't have enough AI credits ({organization.ai_credits_balance} remaining, "
            f"{cost} required). Ask an Owner to add more credits.",
        )


def deduct_ai_credits(organization, cost, reason, actor=None):
    """
    Spends `cost` credits, floored at 0 (never negative - a race
    between check_ai_credits() and this call losing by a hair
    shouldn't leave the ledger showing a negative balance, just an
    exact-zero one). Locks the Organization row for the duration of
    the update (select_for_update) so two concurrent spends can't both
    read the same starting balance and each independently subtract
    from it - the classic lost-update race a bare
    `F("ai_credits_balance") - cost` update alone wouldn't fully guard
    against once AICreditTransaction needs a real `balance_after`
    snapshot to write alongside it.
    """

    if organization is None or cost <= 0:
        return

    from django.db import transaction

    with transaction.atomic():
        locked = Organization.objects.select_for_update().get(pk=organization.pk)
        if locked.ai_credits_balance is None:
            return  # unlimited - nothing to spend down, no ledger row for an untracked balance
        locked.ai_credits_balance = max(0, locked.ai_credits_balance - cost)
        locked.save(update_fields=["ai_credits_balance"])
        AICreditTransaction.objects.create(
            organization=locked, amount=-cost, balance_after=locked.ai_credits_balance,
            reason=reason, actor=actor,
        )
    organization.ai_credits_balance = locked.ai_credits_balance


def add_ai_credits(organization, amount, actor, reason="Credits added by Owner"):
    """
    Owner-only top-up (enforced by the view, not here). Only permitted
    when the org's active Plan allows it (allow_credit_purchase) - the
    caller (organization_ai_credits_view) checks this before calling,
    same "caller enforces the gate, function trusts it" pattern the
    rest of this module already uses. `amount` must be positive; there
    is no separate "remove credits" operation - an Owner corrects an
    over-grant by not spending it, not by a negative top-up, keeping
    every row in the ledger a real event (a top-up or a spend) rather
    than an arbitrary adjustment.
    """

    if amount <= 0:
        raise BillingServiceError("The amount must be a positive number of credits.")

    from django.db import transaction

    with transaction.atomic():
        locked = Organization.objects.select_for_update().get(pk=organization.pk)
        # First-ever top-up is what "activates" metering for this org
        # (see the field's help_text) - a NULL (unlimited, untouched)
        # balance becomes a real number starting from 0, not from NULL.
        locked.ai_credits_balance = (locked.ai_credits_balance or 0) + amount
        locked.save(update_fields=["ai_credits_balance"])
        AICreditTransaction.objects.create(
            organization=locked, amount=amount, balance_after=locked.ai_credits_balance,
            reason=reason, actor=actor,
        )
    # Keeps the CALLER's in-memory instance in sync (same reason
    # deduct_ai_credits() does this) - add_ai_credits()/deduct_ai_credits()
    # both operate on a freshly select_for_update()'d `locked` copy, not
    # the `organization` object a caller already holds; without this, a
    # caller that reads `organization.ai_credits_balance` (or passes the
    # same object into check_ai_credits() right after) would see the
    # stale pre-transaction value.
    organization.ai_credits_balance = locked.ai_credits_balance
    return locked.ai_credits_balance


# ---- Personal-Workspace counterparts of the credit functions above ----

def check_personal_ai_credits(user, cost):
    if user is None:
        return
    profile = user.profile
    if profile.ai_credits_balance is None:
        return
    if profile.ai_credits_balance < cost:
        raise UsageLimitExceeded(
            "ai_credits",
            f"Your personal workspace doesn't have enough AI credits ({profile.ai_credits_balance} remaining, "
            f"{cost} required). Add more credits to continue.",
        )


def deduct_personal_ai_credits(user, cost, reason, actor=None):
    if user is None or cost <= 0:
        return

    from django.db import transaction

    with transaction.atomic():
        locked = UserProfile.objects.select_for_update().get(user_id=user.pk)
        if locked.ai_credits_balance is None:
            return
        locked.ai_credits_balance = max(0, locked.ai_credits_balance - cost)
        locked.save(update_fields=["ai_credits_balance"])
        AICreditTransaction.objects.create(
            user=user, amount=-cost, balance_after=locked.ai_credits_balance,
            reason=reason, actor=actor,
        )
    user.profile.ai_credits_balance = locked.ai_credits_balance


def add_personal_ai_credits(user, amount, actor, reason="Credits added"):
    if amount <= 0:
        raise BillingServiceError("The amount must be a positive number of credits.")

    from django.db import transaction

    with transaction.atomic():
        locked = UserProfile.objects.select_for_update().get(user_id=user.pk)
        locked.ai_credits_balance = (locked.ai_credits_balance or 0) + amount
        locked.save(update_fields=["ai_credits_balance"])
        AICreditTransaction.objects.create(
            user=user, amount=amount, balance_after=locked.ai_credits_balance,
            reason=reason, actor=actor,
        )
    user.profile.ai_credits_balance = locked.ai_credits_balance
    return locked.ai_credits_balance


def _validate_feature_codes(codes):
    """Normalizes to a sorted, de-duplicated list; raises on any code outside FEATURE_CODES
    so a typo in the admin UI can never silently create an unenforceable restriction."""
    unknown = set(codes or []) - set(FEATURE_CODES)
    if unknown:
        raise BillingServiceError(f"Unknown feature code(s): {', '.join(sorted(unknown))}")
    return sorted(set(codes or []))


def org_plan_feature_codes(organization):
    """
    None = unrestricted (every FEATURE_CODES entry included). Personal
    Workspace (organization=None), no Subscription, an inactive Plan,
    or a Plan whose included_features is still the [] default all mean
    "no restriction" - the same "no row = unlimited" contract
    get_organization_usage() already establishes for quantity limits,
    now extended to feature access. Personal Workspace's own feature
    ceiling is a separate function, personal_plan_feature_codes() -
    org_plan_feature_codes(None) staying None (unrestricted) here is
    deliberate backward-compatible behavior for any caller that still
    only ever passes an organization.
    """
    if organization is None:
        return None
    subscription = get_active_subscription(organization)
    if subscription is None or not subscription.plan.is_active:
        return None
    return subscription.plan.included_features or None


def org_has_plan_feature(organization, feature_code):
    codes = org_plan_feature_codes(organization)
    return codes is None or feature_code in codes


def personal_plan_feature_codes(user):
    """The Personal-Workspace counterpart to org_plan_feature_codes() - None = unrestricted (no Personal Plan assigned, or an inactive one, or one with no feature restriction)."""
    if user is None:
        return None
    subscription = get_active_personal_subscription(user)
    if subscription is None or not subscription.plan.is_active:
        return None
    return subscription.plan.included_features or None


def personal_has_plan_feature(user, feature_code):
    codes = personal_plan_feature_codes(user)
    return codes is None or feature_code in codes


def create_plan(
    name, description="", plan_type=Plan.PlanType.COMPANY,
    price=None, currency="USD", billing_interval=Plan.BillingInterval.MONTHLY,
    max_queries_per_month=None, max_ai_task_runs_per_month=None, max_storage_bytes=None, max_seats=None,
    included_credits=0, allow_credit_purchase=True, included_features=None,
    actor=None, request=None,
):
    name = (name or "").strip()
    if not name:
        raise BillingServiceError("A plan name is required.")
    if plan_type not in Plan.PlanType.values:
        raise BillingServiceError(f"Unknown plan_type: {plan_type}")
    if billing_interval not in Plan.BillingInterval.values:
        raise BillingServiceError(f"Unknown billing_interval: {billing_interval}")

    from django.utils.text import slugify
    slug = slugify(name)[:110] or "plan"
    if Plan.objects.filter(slug=slug).exists():
        raise BillingServiceError(f'A plan named "{name}" already exists.')

    plan = Plan.objects.create(
        name=name, slug=slug, description=description, plan_type=plan_type,
        price=price, currency=currency or "USD", billing_interval=billing_interval,
        max_queries_per_month=max_queries_per_month, max_ai_task_runs_per_month=max_ai_task_runs_per_month,
        max_storage_bytes=max_storage_bytes, max_seats=max_seats,
        included_credits=included_credits or 0, allow_credit_purchase=allow_credit_purchase,
        included_features=_validate_feature_codes(included_features),
    )
    log_activity(actor, "billing.plan_created", f'Plan "{plan.name}" created', request=request)
    return plan


def update_plan(plan, actor=None, request=None, **fields):
    if "included_features" in fields:
        fields["included_features"] = _validate_feature_codes(fields["included_features"])

    changed = []
    for field, value in fields.items():
        if getattr(plan, field) != value:
            setattr(plan, field, value)
            changed.append(field)
    if changed:
        plan.save(update_fields=changed + ["updated_at"])
        log_activity(actor, "billing.plan_updated", f'Plan "{plan.name}" updated ({", ".join(changed)})', request=request)
    return plan


def assign_plan(organization, plan, assigned_by, request=None):
    """
    Does NOT reset current_period_start/end on a plan change (only on
    first assignment) - upgrading raises the effective cap
    immediately; downgrading can put an org over-limit for the
    remainder of the CURRENT period only, not a full new cycle. This
    avoids "reset gaming" via rapid plan swaps without any extra logic.
    On first-ever assignment, also immediately refills credits to
    plan.included_credits (see _refill_credits_for_subscription()) -
    an org shouldn't sit at 0/unmetered until the next period boundary
    just because it was assigned mid-period.
    """
    now = timezone.now()
    subscription, created = Subscription.objects.get_or_create(
        organization=organization,
        defaults={"plan": plan, "assigned_by": assigned_by, "current_period_start": now, "current_period_end": _advance_period_end(now, plan)},
    )
    if not created:
        subscription.plan = plan
        subscription.assigned_by = assigned_by
        subscription.save(update_fields=["plan", "assigned_by", "updated_at"])
    if created:
        _refill_credits_for_subscription(subscription, reason="Plan assigned")

    log_org_activity(
        organization, assigned_by, "billing.plan_assigned", f"Plan changed to {plan.name}",
        metadata={"plan_id": plan.id}, request=request,
    )
    return subscription


def unassign_plan(organization, actor, request=None):
    """Deletes the Subscription row - reverts the org to unlimited, the same 'no row = unlimited' default a brand new organization already starts in."""
    deleted, _ = Subscription.objects.filter(organization=organization).delete()
    if deleted:
        log_org_activity(
            organization, actor, "billing.plan_unassigned", "Plan removed - organization reverted to unlimited.",
            request=request,
        )


def assign_personal_plan(user, plan, assigned_by, request=None):
    """The Personal-Workspace counterpart to assign_plan() - same first-assignment credit refill, no OrganizationAuditLog (Personal Workspace has none); logged to the platform ActivityLog instead."""
    now = timezone.now()
    subscription, created = Subscription.objects.get_or_create(
        user=user,
        defaults={"plan": plan, "assigned_by": assigned_by, "current_period_start": now, "current_period_end": _advance_period_end(now, plan)},
    )
    if not created:
        subscription.plan = plan
        subscription.assigned_by = assigned_by
        subscription.save(update_fields=["plan", "assigned_by", "updated_at"])
    if created:
        _refill_credits_for_subscription(subscription, reason="Plan assigned")

    log_activity(assigned_by, "billing.personal_plan_assigned", f"{user.username}'s personal plan changed to {plan.name}", request=request)
    return subscription


def unassign_personal_plan(user, actor, request=None):
    deleted, _ = Subscription.objects.filter(user=user).delete()
    if deleted:
        log_activity(actor, "billing.personal_plan_unassigned", f"{user.username}'s personal plan removed - reverted to unlimited.", request=request)


# ============================================================
# Plan change requests - the Owner/Personal-user self-service path in
# front of assign_plan()/assign_personal_plan() above. See
# PlanChangeRequest's model docstring for the full rationale.
# ============================================================

def request_plan_change(plan, requested_by, organization=None, user=None, request=None):
    if (organization is None) == (user is None):
        raise BillingServiceError("Exactly one of organization or user is required.")
    if plan.plan_type == Plan.PlanType.COMPANY and organization is None:
        raise BillingServiceError("This plan is a Company plan and can't be requested for a Personal Workspace.")
    if plan.plan_type == Plan.PlanType.PERSONAL and user is None:
        raise BillingServiceError("This plan is a Personal plan and can't be requested for a Company organization.")
    if not plan.is_active:
        raise BillingServiceError("This plan is no longer available.")

    existing_pending = PlanChangeRequest.objects.filter(
        organization=organization, user=user, status=PlanChangeRequest.Status.PENDING,
    ).exists()
    if existing_pending:
        raise BillingServiceError("There's already a pending plan request - wait for it to be reviewed before requesting another.")

    plan_request = PlanChangeRequest.objects.create(
        organization=organization, user=user, requested_plan=plan, requested_by=requested_by,
    )
    if organization is not None:
        log_org_activity(organization, requested_by, "billing.plan_requested", f"Requested plan change to {plan.name}", metadata={"plan_id": plan.id}, request=request)
    else:
        log_activity(requested_by, "billing.personal_plan_requested", f"Requested personal plan change to {plan.name}", request=request)
    return plan_request


def approve_plan_request(plan_request, actor, request=None):
    if plan_request.status != PlanChangeRequest.Status.PENDING:
        raise BillingServiceError("This request has already been reviewed.")

    if plan_request.organization_id:
        assign_plan(plan_request.organization, plan_request.requested_plan, actor, request=request)
    else:
        assign_personal_plan(plan_request.user, plan_request.requested_plan, actor, request=request)

    plan_request.status = PlanChangeRequest.Status.APPROVED
    plan_request.reviewed_by = actor
    plan_request.reviewed_at = timezone.now()
    plan_request.save(update_fields=["status", "reviewed_by", "reviewed_at"])
    return plan_request


def check_query_limit_for(organization, user):
    """Dispatches to the Company or Personal check depending on which workspace `organization` names (None = Personal, scoped to `user`) - the one place enforcement call sites (ask_views.py, ...) branch, instead of repeating `if organization is not None: ... else: ...` at every site."""
    if organization is not None:
        check_query_limit(organization)
    else:
        check_personal_query_limit(user)


def check_ai_task_limit_for(organization, user):
    if organization is not None:
        check_ai_task_limit(organization)
    else:
        check_personal_ai_task_limit(user)


def check_storage_limit_for(organization, user, additional_bytes):
    if organization is not None:
        check_storage_limit(organization, additional_bytes)
    else:
        check_personal_storage_limit(user, additional_bytes)


def check_ai_credits_for(organization, user, cost):
    if organization is not None:
        check_ai_credits(organization, cost)
    else:
        check_personal_ai_credits(user, cost)


def deduct_ai_credits_for(organization, user, cost, reason, actor=None):
    if organization is not None:
        deduct_ai_credits(organization, cost, reason, actor=actor)
    else:
        deduct_personal_ai_credits(user, cost, reason, actor=actor)


def reject_plan_request(plan_request, actor, note="", request=None):
    if plan_request.status != PlanChangeRequest.Status.PENDING:
        raise BillingServiceError("This request has already been reviewed.")

    plan_request.status = PlanChangeRequest.Status.REJECTED
    plan_request.reviewed_by = actor
    plan_request.reviewed_at = timezone.now()
    plan_request.admin_note = note or ""
    plan_request.save(update_fields=["status", "reviewed_by", "reviewed_at", "admin_note"])
    return plan_request
