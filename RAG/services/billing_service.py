"""
Billing / Plans / Usage Limits

Internal bookkeeping only - no real payment processor, no card
charging. A Super Admin manually assigns a Plan to an Organization
(billing_views.py); this module enforces usage limits with a hard
block once exceeded, and reports current usage. An Owner can also
REQUEST a plan change themselves (request_plan_change()), which stays
Pending until a Platform Admin approves it (approve_plan_request()) -
direct assignment by a Platform Admin (assign_plan()) is a separate,
instant, admin-initiated path that skips the request queue entirely.

Two independent enforcement axes (deliberately not merged into one -
see the "Billing / Plans / Usage Limits" comment block at the top of
models.py for the full rationale):

  - AI CREDITS (Organization.ai_credits_balance) are the organization's
    SOLE spend-metering currency now. Every question asked / AI Task
    run costs credits (check_ai_credits()/deduct_ai_credits()), and the
    instant ANY Plan is assigned, credits become strictly governed by
    that Plan's `included_credits` (refilled every billing period - see
    _advance_period()). Plan.max_queries_per_month/
    max_ai_task_runs_per_month are informational quota display only -
    check_query_limit()/check_ai_task_limit() were removed along with
    this redesign, since a second, independent count-based gate on the
    exact same action (asking a question, running an AI Task) just
    meant an org could be blocked by whichever gate was stricter while
    the other sat unused - not a meaningful extra control, just
    confusing double metering.
  - PER-MEMBER quantity caps (OrganizationMembership.
    max_queries_per_month/max_ai_task_runs_per_month) are a SEPARATE,
    Owner-set, count-based ceiling on one specific member's usage -
    check_member_query_limit()/check_member_ai_task_limit() below.
    Unlike credits, these remain a real hard block independent of the
    organization's shared credit balance: an Owner narrowing one
    member's monthly allowance doesn't touch what the rest of the
    organization can spend.

No Subscription row for an organization means unlimited - the caller
(get_active_subscription) never has to special-case "new organization"
vs. "explicitly unlimited organization", and get_organization_usage()
returns immediately (no query) when that's the case. This is the same
"absence means today's behavior, nothing breaks until someone opts in"
rollout convention this codebase uses elsewhere for additive, nullable
rollout fields.

(Personal Workspace billing - `user=`-keyed Subscriptions/credits -
was removed along with the Personal Workspace account type; the
`_for`-suffixed dispatchers below (check_query_limit_for, ...) are
kept only so ask_views.py/ai_tasks_views.py/documents_views.py don't
need to call check_member_query_limit() etc. directly - every
workspace is an Organization now.)

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


def get_active_subscription(organization):
    """None for an org with no Subscription row (or no organization at all) - both mean unlimited."""
    if organization is None:
        return None
    return Subscription.objects.select_related("plan").filter(organization=organization).first()


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
    (_advance_period) and on first-ever Plan assignment (assign_plan)
    alike, so a brand new Subscription starts funded immediately
    rather than waiting for the next boundary.
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


def _current_period_window(organization):
    """
    The (start, end) window per-member caps are counted against.
    Reuses the org's own active Subscription period when one exists,
    so a member's window always matches the organization's own -
    otherwise falls back to the current calendar month, since an
    Owner's per-member cap is that Owner's own choice, independent of
    whether a Plan is assigned at all (unlike credits, which have
    nothing to meter without a Plan's included_credits to draw from).
    """
    subscription = get_active_subscription(organization)
    if subscription is not None:
        return _advance_period(subscription)
    now = timezone.now()
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return start, _add_one_month(start)


def get_member_usage(organization, user, membership=None):
    """
    Per-member usage counts for the current period window - the
    sub-quota counterpart to get_organization_usage(). Returns None
    when `user` has no active membership in `organization` (nothing to
    report). Unlike get_organization_usage(), this is not cached -
    it's only read on a member-detail/roster view and on the
    check_member_*_limit() hard-block path below, nowhere near
    per-request-hot enough to need it.
    """
    if membership is None:
        from .org_permission_service import get_org_membership
        membership = get_org_membership(user, organization)
    if membership is None:
        return None

    period_start, period_end = _current_period_window(organization)
    return {
        "period_start": period_start,
        "period_end": period_end,
        "max_queries_per_month": membership.max_queries_per_month,
        "max_ai_task_runs_per_month": membership.max_ai_task_runs_per_month,
        "queries_used": QueryLog.objects.filter(
            organization=organization, user=user, created_at__gte=period_start, created_at__lt=period_end,
        ).count(),
        "ai_task_runs_used": AITaskRun.objects.filter(
            organization=organization, user=user, created_at__gte=period_start, created_at__lt=period_end,
        ).count(),
    }


def _check_member(organization, user, used_key, limit_key, limit_type, message_noun):
    if organization is None:
        return
    from .org_permission_service import get_org_membership
    membership = get_org_membership(user, organization)
    if membership is None:
        return
    limit = getattr(membership, limit_key)
    if limit is None:
        return
    usage = get_member_usage(organization, user, membership=membership)
    if usage[used_key] >= limit:
        raise UsageLimitExceeded(
            limit_type,
            f"You've reached your personal monthly {message_noun} limit ({limit}) set by your "
            f"organization's Owner. Ask an Owner to raise it.",
        )


def check_member_query_limit(organization, user):
    _check_member(organization, user, "queries_used", "max_queries_per_month", "member_queries", "AI query")


def check_member_ai_task_limit(organization, user):
    _check_member(organization, user, "ai_task_runs_used", "max_ai_task_runs_per_month", "member_ai_task_runs", "AI Task run")


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


def _validate_feature_codes(codes):
    """Normalizes to a sorted, de-duplicated list; raises on any code outside FEATURE_CODES
    so a typo in the admin UI can never silently create an unenforceable restriction."""
    unknown = set(codes or []) - set(FEATURE_CODES)
    if unknown:
        raise BillingServiceError(f"Unknown feature code(s): {', '.join(sorted(unknown))}")
    return sorted(set(codes or []))


def org_plan_feature_codes(organization):
    """
    None = unrestricted (every FEATURE_CODES entry included). No
    organization, no Subscription, an inactive Plan, or a Plan whose
    included_features is still the [] default all mean "no
    restriction" - the same "no row = unlimited" contract
    get_organization_usage() already establishes for quantity limits,
    now extended to feature access.
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


def delete_plan(plan, actor=None, request=None):
    """
    Hard-deletes a Plan that has never been assigned or requested.
    Subscription.plan and PlanChangeRequest.requested_plan are both
    on_delete=PROTECT (see their docstrings), so Django itself refuses
    this at the DB level the instant any organization/user has ever
    been on this Plan or requested it - deletion is only for a Plan
    created by mistake or never adopted. A Plan already in real use
    should be deactivated instead (update_plan(plan, is_active=False)),
    which every existing subscriber keeps working under.
    """

    from django.db.models import ProtectedError

    name = plan.name
    try:
        plan.delete()
    except ProtectedError:
        raise BillingServiceError(
            f'"{name}" can\'t be deleted - it has active subscriptions or plan-change history. Deactivate it instead.'
        )
    log_activity(actor, "billing.plan_deleted", f'Plan "{name}" deleted', request=request)


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


FREE_PLAN_SLUG = "free"


def get_or_create_free_plan():
    """
    The built-in, zero-configuration Company plan every new organization
    is assigned to at creation (organization_service.create_organization()) -
    a brand-new company should start on a real, working plan rather than
    the old "no Subscription row = unlimited" default, so usage limits
    are meaningful from day one instead of only after a Super Admin
    manually assigns one. Idempotent (get_or_create by slug), so it's
    safe to call both from the seed_billing_plans management command
    and from organization creation itself - an environment where nobody
    ever ran the seed command still gets a working Free plan on the
    very first company signup, not a missing-plan error. Re-running
    never overwrites an already-existing Free plan's live settings
    (an admin may have since edited its limits) - `defaults` only apply
    on first creation, same convention seed_organization_types.py uses.
    """
    plan, _ = Plan.objects.get_or_create(
        slug=FREE_PLAN_SLUG,
        defaults={
            "name": "Free",
            "description": "Default plan for every new company - no cost, enough to get started.",
            "plan_type": Plan.PlanType.COMPANY,
            "price": None,
            "included_credits": 50,
            "max_queries_per_month": 100,
            "max_ai_task_runs_per_month": 20,
            "max_seats": 5,
            "is_active": True,
        },
    )
    return plan


# ============================================================
# Plan change requests - the Owner self-service path in front of
# assign_plan() above. See PlanChangeRequest's model docstring for the
# full rationale.
# ============================================================

def request_plan_change(plan, requested_by, organization, request=None):
    if not plan.is_active:
        raise BillingServiceError("This plan is no longer available.")

    existing_pending = PlanChangeRequest.objects.filter(
        organization=organization, status=PlanChangeRequest.Status.PENDING,
    ).exists()
    if existing_pending:
        raise BillingServiceError("There's already a pending plan request - wait for it to be reviewed before requesting another.")

    plan_request = PlanChangeRequest.objects.create(
        organization=organization, requested_plan=plan, requested_by=requested_by,
    )
    log_org_activity(organization, requested_by, "billing.plan_requested", f"Requested plan change to {plan.name}", metadata={"plan_id": plan.id}, request=request)
    return plan_request


def approve_plan_request(plan_request, actor, request=None):
    if plan_request.status != PlanChangeRequest.Status.PENDING:
        raise BillingServiceError("This request has already been reviewed.")

    assign_plan(plan_request.organization, plan_request.requested_plan, actor, request=request)

    plan_request.status = PlanChangeRequest.Status.APPROVED
    plan_request.reviewed_by = actor
    plan_request.reviewed_at = timezone.now()
    plan_request.save(update_fields=["status", "reviewed_by", "reviewed_at"])
    return plan_request


def check_query_limit_for(organization, user):
    """Every workspace is now an Organization - kept as the one place enforcement call sites (ask_views.py, ...) go through, rather than calling check_member_query_limit() directly, so a future workspace type doesn't require touching every call site again. Org-wide usage is metered by credits alone (check_ai_credits_for, called separately by every caller) - this is only the per-member sub-quota gate."""
    check_member_query_limit(organization, user)


def check_ai_task_limit_for(organization, user):
    check_member_ai_task_limit(organization, user)


def check_storage_limit_for(organization, user, additional_bytes):
    check_storage_limit(organization, additional_bytes)


def check_ai_credits_for(organization, user, cost):
    check_ai_credits(organization, cost)


def deduct_ai_credits_for(organization, user, cost, reason, actor=None):
    deduct_ai_credits(organization, cost, reason, actor=actor)


def reject_plan_request(plan_request, actor, note="", request=None):
    if plan_request.status != PlanChangeRequest.Status.PENDING:
        raise BillingServiceError("This request has already been reviewed.")

    plan_request.status = PlanChangeRequest.Status.REJECTED
    plan_request.reviewed_by = actor
    plan_request.reviewed_at = timezone.now()
    plan_request.admin_note = note or ""
    plan_request.save(update_fields=["status", "reviewed_by", "reviewed_at", "admin_note"])
    return plan_request
