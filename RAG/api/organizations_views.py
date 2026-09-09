"""
Organization endpoints for the React SPA - workspace switching, org
profile, member management, invitations, org-scoped audit logs, and
(gated on the platform "organizations.*" permissions - see
org_permission_service's module docstring) Super Admin oversight of
every organization on the platform.

Every org-scoped view (one whose URL includes <slug:org_slug>) is
gated with HasOrgPermission(...) (RAG/api/permissions.py), which
re-derives membership from OrganizationMembership on every request and
attaches the trusted request.organization / request.org_membership -
no view body below ever trusts org_slug beyond "which organization is
being asked about", and no view body re-implements that resolution
itself.
"""

from django.db.models import Sum
from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response

from ..models import ORG_ROLE_OWNER, AITaskRun, Document, Organization, OrganizationInvitation, OrganizationMembership, OrganizationType, UserProfile
from ..services import billing_service, observability_service, org_member_registration_service, org_membership_service
from ..services.billing_service import UsageLimitExceeded
from ..services.knowledge_service import get_knowledge_overview
from ..services.org_invitation_service import InvitationError, accept_invitation, create_invitation, list_invitations, revoke_invitation
from ..services.org_member_feature_service import FeatureAccessError, get_member_feature_access, set_member_disabled_features
from ..services.org_member_limits_service import MemberLimitError, set_member_usage_limits
from ..services.org_membership_service import MembershipError
from ..services.org_permission_service import (
    can_actor_assign_org_role,
    get_assignable_org_roles,
    get_org_permission_codenames,
    get_user_organizations,
    user_has_org_permission,
)
from ..services.organization_service import OrganizationServiceError, create_organization, delete_organization, reactivate_organization, suspend_organization, update_organization
from ..services.permission_service import user_has_permission
from ..services.stats_service import (
    get_document_type_breakdown,
    get_documents_over_time,
    get_kpi_trends,
    get_member_activity_breakdown,
    get_my_activity_summary,
    get_recent_documents_table,
)
from ..utils.formatting import format_bytes
from .permissions import HasOrgPermission, HasPagePermission


def _serialize_org_type(t):
    return {"slug": t.slug, "name": t.name}


_UNSET = object()  # distinguishes "no batch value was passed" from "the batch value is genuinely None/0" (an owner-less org, or a zero-member one)


def _organization_owner_label(organization):
    """The Owner's display name for a SINGLE organization - derived from OrganizationMembership rather than a stored field (see Organization's docstring on why there's no `owner` FK). Only used for a lone-org lookup; a list of many organizations must use _owner_labels_by_org_id() instead, or this becomes one query per row."""
    owner_membership = (
        organization.memberships
        .filter(role=ORG_ROLE_OWNER, status=OrganizationMembership.Status.ACTIVE)
        .select_related("user")
        .first()
    )
    if owner_membership is None:
        return None
    return owner_membership.user.get_full_name() or owner_membership.user.username


def _owner_labels_by_org_id(organizations):
    """Batch counterpart to _organization_owner_label() - one query for every organization's Owner instead of one query per organization, for a platform-wide listing (see platform_organizations_view)."""
    owner_memberships = (
        OrganizationMembership.objects
        .filter(organization__in=organizations, role=ORG_ROLE_OWNER, status=OrganizationMembership.Status.ACTIVE)
        .select_related("user")
    )
    return {m.organization_id: (m.user.get_full_name() or m.user.username) for m in owner_memberships}


def _member_counts_by_org_id(organizations):
    """Batch counterpart to the per-row `organization.memberships.filter(...).count()` - one grouped query for every organization's active member count instead of one COUNT per organization."""
    from django.db.models import Count, Q

    rows = (
        Organization.objects.filter(id__in=[o.id for o in organizations])
        .annotate(_active_member_count=Count("memberships", filter=Q(memberships__status=OrganizationMembership.Status.ACTIVE)))
        .values("id", "_active_member_count")
    )
    return {row["id"]: row["_active_member_count"] for row in rows}


def _serialize_organization(organization, membership=None, owner_label=_UNSET, member_count=_UNSET):
    data = {
        "id": organization.id,
        "slug": organization.slug,
        "name": organization.name,
        "org_type": organization.org_type.slug,
        "org_type_name": organization.org_type.name,
        "description": organization.description,
        "logo_url": organization.logo.url if organization.logo else None,
        "website": organization.website,
        "industry": organization.industry,
        "size": organization.size,
        "contact_email": organization.contact_email,
        "contact_phone": organization.contact_phone,
        "status": organization.status,
        "created_at": organization.created_at,
        "member_count": (
            member_count if member_count is not _UNSET
            else organization.memberships.filter(status=OrganizationMembership.Status.ACTIVE).count()
        ),
        "owner": owner_label if owner_label is not _UNSET else _organization_owner_label(organization),
    }
    if membership is not None:
        data["my_role"] = membership.role
        data["my_permissions"] = get_org_permission_codenames(membership.user, organization)
    return data


def _serialize_member(membership):
    user = membership.user
    return {
        "membership_id": membership.id,
        "user_id": user.id,
        "username": user.username,
        "full_name": user.get_full_name(),
        "email": user.email,
        "avatar_url": user.profile.avatar.url if getattr(user, "profile", None) and user.profile.avatar else None,
        "role": membership.role,
        "status": membership.status,
        "joined_at": membership.joined_at,
        "disabled_features": membership.disabled_features,
        "feature_access": get_member_feature_access(membership.organization, membership),
        "max_queries_per_month": membership.max_queries_per_month,
        "max_ai_task_runs_per_month": membership.max_ai_task_runs_per_month,
        "usage": billing_service.get_member_usage(membership.organization, user, membership=membership),
    }


def _serialize_invitation(invitation):
    return {
        "id": invitation.id,
        "email": invitation.email,
        "role": invitation.role,
        "status": invitation.status,
        "invited_by": invitation.invited_by.username if invitation.invited_by_id else None,
        "created_at": invitation.created_at,
        "expires_at": invitation.expires_at,
    }


def _serialize_audit_log(entry, include_ip=False):
    data = {
        "id": entry.id,
        "actor": entry.actor.username if entry.actor_id else None,
        "action": entry.action,
        "description": entry.description,
        "metadata": entry.metadata,
        "created_at": entry.created_at,
    }
    # IP address is only meaningful to someone who can already see this
    # entire log (HasOrgPermission("audit_logs.view"), ADMIN+ - the
    # view below is the only caller), so there's no separate permission
    # tier to gate it behind here; it's just kept out of the default
    # shape until a caller actually asks, matching how sparingly the
    # rest of the SPA surfaces raw IPs (activity.view_ip_location is a
    # SENSITIVE_PERMISSIONS entry precisely because of this).
    if include_ip:
        data["ip_address"] = entry.ip_address
    return data


# ============================================================
# Workspace switcher / org CRUD
# ============================================================

@api_view(["GET"])
@permission_classes([AllowAny])
def organization_types_view(request):
    """Public - a static reference catalog (company/startup/enterprise/...), not sensitive, and needed by the signup form's Company registration step before the visitor has an account at all."""
    return Response({"organization_types": [_serialize_org_type(t) for t in OrganizationType.objects.filter(is_active=True)]})


@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def organizations_view(request):
    """GET: every ACTIVE organization the current user belongs to (the Company-to-Company workspace switcher's data - see fetch_session_payload's account_type). POST: register an additional organization for an existing COMPANY account (the creator becomes its Owner - see organization_service.create_organization). A PERSONAL account can never reach this: the Personal-vs-Company decision is made once at signup (RAG.api.auth_views.signup_view), never through a "create my first organization" self-service action after the fact."""

    if request.method == "GET":
        memberships = get_user_organizations(request.user)
        return Response({
            "organizations": [_serialize_organization(m.organization, m) for m in memberships],
        })

    if getattr(getattr(request.user, "profile", None), "account_type", UserProfile.AccountType.PERSONAL) != UserProfile.AccountType.COMPANY:
        return Response({"error": "Personal accounts can't create organizations. Sign up as a Company to create one."}, status=403)

    org_type = OrganizationType.objects.filter(slug=request.data.get("org_type"), is_active=True).first()
    if not org_type:
        return Response({"error": "Invalid organization type."}, status=400)

    name = (request.data.get("name") or "").strip()
    if not name:
        return Response({"error": "Organization name is required."}, status=400)

    try:
        organization = create_organization(
            name=name,
            org_type=org_type,
            created_by=request.user,
            slug=request.data.get("slug"),
            description=request.data.get("description", ""),
            website=request.data.get("website", ""),
            industry=request.data.get("industry", ""),
            contact_email=request.data.get("contact_email", ""),
            contact_phone=request.data.get("contact_phone", ""),
        )
    except OrganizationServiceError as e:
        return Response({"error": str(e)}, status=400)

    membership = OrganizationMembership.objects.get(organization=organization, user=request.user)
    return Response(_serialize_organization(organization, membership), status=201)


@api_view(["GET", "PATCH"])
@permission_classes([HasOrgPermission("organization.view")])
def organization_detail_view(request, org_slug):
    if request.method == "GET":
        return Response(_serialize_organization(request.organization, request.org_membership))

    if not user_has_org_permission(request.user, request.organization, "organization.update"):
        return Response({"error": "You don't have permission to update this organization."}, status=403)

    editable_fields = ("name", "description", "website", "industry", "contact_email", "contact_phone")
    fields = {k: request.data[k] for k in editable_fields if k in request.data}

    if "org_type" in request.data:
        org_type = OrganizationType.objects.filter(slug=request.data["org_type"], is_active=True).first()
        if not org_type:
            return Response({"error": "Invalid organization type."}, status=400)
        fields["org_type"] = org_type

    update_organization(request.organization, request.user, request=request, **fields)
    return Response(_serialize_organization(request.organization, request.org_membership))


@api_view(["GET"])
@permission_classes([HasOrgPermission("organization.view")])
def organization_stats_view(request, org_slug):
    """
    Usage snapshot for the Organization Overview page - document/
    storage/AI Task/member/search/activity counts scoped strictly to
    `request.organization`, the URL-verified org HasOrgPermission
    resolved (never the X-Organization-Slug header). Passes that same
    `organization` into stats_service's now workspace-aware aggregates
    (get_kpi_trends/get_documents_over_time - see their
    _workspace_scope() helper) rather than hand-rolling a second set of
    trend queries - a member can be viewing this organization's
    Overview page while a *different* workspace is active in the
    sidebar, and this page must always describe the organization named
    in its own URL, never whichever one the header happens to carry.
    """

    from datetime import timedelta

    from django.utils import timezone

    from ..models import QueryLog

    documents = Document.objects.filter(organization=request.organization)
    query_logs = QueryLog.objects.filter(organization=request.organization)

    active_members_7d = (
        query_logs.filter(created_at__gte=timezone.now() - timedelta(days=7))
        .values("user_id").distinct().count()
    )

    member_qs = OrganizationMembership.objects.filter(
        organization=request.organization, status=OrganizationMembership.Status.ACTIVE,
    )

    return Response({
        "document_count": documents.count(),
        "total_storage": format_bytes(documents.aggregate(total=Sum("file_size"))["total"] or 0),
        "ai_task_run_count": AITaskRun.objects.filter(organization=request.organization).count(),
        "topic_count": get_knowledge_overview(request.user, organization=request.organization)["total_entities"],
        "member_count": request.organization.memberships.filter(status=OrganizationMembership.Status.ACTIVE).count(),
        "question_count": query_logs.count(),
        "active_members_7d": active_members_7d,
        "documents_over_time": get_documents_over_time(request.user, days=14, organization=request.organization),
        "kpi_trends": get_kpi_trends(request.user, organization=request.organization),
        "document_types": get_document_type_breakdown(request.user, organization=request.organization),
        "recent_documents_table": [
            {
                "id": row["id"],
                "title": row["title"],
                "owner": row["owner"],
                "file_type": row["file_type"],
                "chunk_count": row["chunk_count"],
                "size": row["size"],
                "size_bytes": row["size_bytes"],
                "uploaded_at": row["uploaded_at"].isoformat(),
                "status": row["status"],
            }
            for row in get_recent_documents_table(request.user, organization=request.organization)
        ],
        "ai_credits_balance": request.organization.ai_credits_balance,
        "recent_queries": [_serialize_query_log_metadata(q) for q in query_logs.select_related("user").order_by("-created_at")[:10]],
        "ai_performance": observability_service.get_performance_summary({"organization": request.organization}),
        "member_activity": get_member_activity_breakdown(request.organization),
        "my_activity": get_my_activity_summary(request.user, request.organization),
        # Same {feature_code: bool} shape dashboard_view exposes -
        # lets Company Owner Overview hide a KPI/chart card the org's
        # own Plan doesn't include, same as a plain Member's Overview
        # already does (Owner is still bounded by the Plan ceiling
        # here, just never by a per-member override - see
        # get_member_feature_access()'s docstring).
        "feature_access": get_member_feature_access(request.organization, request.org_membership),
        "feature_access_summary": {
            "plan_feature_codes": billing_service.org_plan_feature_codes(request.organization),
            # Excludes Owner rows: disabled_features is never enforced for an
            # Owner (has_feature_access()/get_member_feature_access()'s "an
            # Owner is never restricted" contract), so a stale list left over
            # from before a promotion shouldn't count as a real override here.
            "members_with_overrides": member_qs.exclude(role=OrganizationMembership.Role.OWNER).exclude(disabled_features=[]).count(),
        },
    })


def _serialize_query_log_metadata(log):
    """Metadata only - deliberately never `question`/`answer`. See organization_queries_view's docstring for why an Owner's visibility into their own organization's query activity stops at 'who asked something, when, how confident/fast' and never reaches the actual text."""

    return {
        "id": log.id,
        "user": log.user.username if log.user_id else None,
        "confidence": log.confidence,
        "response_time_ms": log.response_time_ms,
        "search_method": log.search_method,
        "created_at": log.created_at,
    }


@api_view(["GET"])
@permission_classes([HasOrgPermission("queries.view")])
def organization_queries_view(request, org_slug):
    """
    Owner-only visibility into an organization's Ask AI activity -
    logs, never content. `queries.view` grants seeing THAT a question
    was asked (who, when, how confident/fast the answer was, which
    retrieval method served it) - the actual question/answer text is
    never serialized here at all, not merely hidden client-side,
    mirroring the platform RBAC precedent `queries.view_content` sets
    (a SENSITIVE_PERMISSIONS entry Super Admin itself doesn't hold -
    see permission_service.py) for the exact same "metadata yes,
    content no" split. Paginated newest-first; no filter params yet -
    add them the same way organization_audit_logs_view's ?action=/
    ?actor= did if this needs to scale past a few pages of scrolling.
    """

    from ..models import QueryLog

    logs = QueryLog.objects.filter(organization=request.organization).select_related("user").order_by("-created_at")

    from django.core.paginator import Paginator
    try:
        page_number = int(request.query_params.get("page", 1))
    except (TypeError, ValueError):
        page_number = 1
    paginator = Paginator(logs, 50)
    page = paginator.get_page(page_number)

    return Response({
        "queries": [_serialize_query_log_metadata(q) for q in page.object_list],
        "page": page.number,
        "num_pages": paginator.num_pages,
        "count": paginator.count,
        "has_next": page.has_next(),
        "has_previous": page.has_previous(),
    })


@api_view(["GET", "POST"])
@permission_classes([HasOrgPermission("ai_credits.view")])
def organization_ai_credits_view(request, org_slug):
    """
    GET: current balance + recent ledger entries (AICreditTransaction -
    both top-ups and spends, newest first). POST: add credits - gated
    by the same `ai_credits.view` codename as the GET, since
    ai_credits.manage/ai_credits.view are both OWNER-only anyway (see
    ORG_PERMISSION_MIN_ROLE) and a real second permission tier here
    would be a distinction with no difference until this codebase ever
    grows a role between Member and Owner again.
    """

    if request.method == "POST":
        subscription = billing_service.get_active_subscription(request.organization)
        if subscription is None or not subscription.plan.allow_credit_purchase:
            return Response({"error": "This organization's current plan doesn't allow purchasing extra credits."}, status=400)
        try:
            amount = int(request.data.get("amount"))
        except (TypeError, ValueError):
            return Response({"error": "A whole number of credits is required."}, status=400)
        try:
            billing_service.add_ai_credits(request.organization, amount, request.user)
        except billing_service.BillingServiceError as e:
            return Response({"error": str(e)}, status=400)
        request.organization.refresh_from_db(fields=["ai_credits_balance"])

    transactions = request.organization.ai_credit_transactions.select_related("actor")[:50]

    return Response({
        "balance": request.organization.ai_credits_balance,
        "transactions": [
            {
                "id": t.id,
                "amount": t.amount,
                "balance_after": t.balance_after,
                "reason": t.reason,
                "actor": t.actor.username if t.actor_id else None,
                "created_at": t.created_at,
            }
            for t in transactions
        ],
    })


# ============================================================
# Members
# ============================================================

@api_view(["GET"])
@permission_classes([HasOrgPermission("members.view")])
def organization_members_view(request, org_slug):
    members = org_membership_service.list_members(request.organization)
    return Response({
        "members": [_serialize_member(m) for m in members],
        "assignable_roles": get_assignable_org_roles(request.org_membership.role),
        "current_user_id": request.user.id,
    })


@api_view(["POST"])
@permission_classes([HasOrgPermission("members.invite")])
def organization_member_register_view(request, org_slug):
    """
    The Owner-only way a company gains a new member (2026-09-06,
    replacing the email-invitation-link flow - see
    org_member_registration_service.py's module docstring for the
    full rationale). Takes a name + email, returns the created (or
    reactivated) membership; the generated username/password are never
    part of this response - they go out exactly once, by email, from
    the background task register_company_member() dispatches.
    """

    try:
        membership = org_member_registration_service.register_company_member(
            request.organization,
            request.data.get("full_name", ""),
            request.data.get("email", ""),
            request.user,
            request=request,
        )
    except org_member_registration_service.MemberRegistrationError as e:
        return Response({"error": str(e)}, status=400)
    except UsageLimitExceeded as e:
        return Response({"error": str(e), "code": "usage_limit_exceeded", "limit_type": e.limit_type}, status=402)

    return Response(_serialize_member(membership), status=201)


@api_view(["POST"])
@permission_classes([HasOrgPermission("members.update_role")])
def organization_member_action_view(request, org_slug):
    """
    update_role/remove/suspend/reactivate/update_features/update_limits
    all mutate another member's standing in the organization, so this
    whole view is gated at OWNER rank (members.update_role's minimum
    in ORG_PERMISSION_MIN_ROLE - members.remove shares that same
    minimum, and the other four actions have no codename of their own,
    so this is the one gate covering all six - and since Member holds
    NO org-management codename at all, Owner is the only rank that can
    ever ask to be checked against). org_membership_service.
    can_actor_manage_org_member()'s rank-relative check still runs
    below as defense-in-depth.
    """

    action = request.data.get("action")
    target = get_object_or_404(OrganizationMembership, id=request.data.get("membership_id"), organization=request.organization)

    try:
        if action == "update_role":
            new_role = request.data.get("role")
            if new_role not in dict(OrganizationMembership.Role.choices):
                return Response({"error": "Invalid role."}, status=400)
            org_membership_service.update_member_role(request.organization, request.org_membership, target, new_role, request=request)
        elif action == "remove":
            org_membership_service.remove_member(request.organization, request.org_membership, target, request=request)
        elif action in ("suspend", "reactivate"):
            status_value = OrganizationMembership.Status.SUSPENDED if action == "suspend" else OrganizationMembership.Status.ACTIVE
            org_membership_service.set_member_status(request.organization, request.org_membership, target, status_value, request=request)
        elif action == "update_features":
            set_member_disabled_features(
                request.organization, request.org_membership, target,
                request.data.get("disabled_features", []), request=request,
            )
        elif action == "update_limits":
            set_member_usage_limits(
                request.organization, request.org_membership, target,
                request.data.get("max_queries_per_month"), request.data.get("max_ai_task_runs_per_month"),
                request=request,
            )
        else:
            return Response({"error": "Unknown action."}, status=400)
    except MembershipError as e:
        return Response({"error": str(e)}, status=403)
    except FeatureAccessError as e:
        return Response({"error": str(e)}, status=400)
    except MemberLimitError as e:
        return Response({"error": str(e)}, status=400)

    return Response({"ok": True})


# ============================================================
# Invitations
# ============================================================

@api_view(["GET", "POST"])
@permission_classes([HasOrgPermission("members.invite")])
def organization_invitations_view(request, org_slug):
    if request.method == "GET":
        invitations = list_invitations(request.organization, status=request.query_params.get("status"))
        return Response({"invitations": [_serialize_invitation(i) for i in invitations]})

    email = (request.data.get("email") or "").strip().lower()
    role = request.data.get("role", OrganizationMembership.Role.MEMBER)

    if not email:
        return Response({"error": "Email is required."}, status=400)
    if role not in dict(OrganizationMembership.Role.choices):
        return Response({"error": "Invalid role."}, status=400)
    if not can_actor_assign_org_role(request.org_membership.role, role):
        return Response({"error": "You don't have permission to invite someone at that role."}, status=403)

    try:
        invitation = create_invitation(request.organization, email, role, request.user, request=request)
    except UsageLimitExceeded as e:
        return Response({"error": str(e), "code": "usage_limit_exceeded", "limit_type": e.limit_type}, status=402)

    from ..services import task_runner
    from ..tasks import send_org_invitation_email_task
    task_runner.submit(send_org_invitation_email_task, invitation.id)

    return Response(_serialize_invitation(invitation), status=201)


@api_view(["POST"])
@permission_classes([HasOrgPermission("members.invite")])
def organization_invitation_revoke_view(request, org_slug, invitation_id):
    invitation = get_object_or_404(OrganizationInvitation, id=invitation_id, organization=request.organization)
    revoke_invitation(invitation, request.user, request=request)
    return Response({"ok": True})


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def organization_invitation_accept_view(request):
    """Not org-scoped by URL (the token itself identifies the organization) and deliberately NOT gated by HasOrgPermission - the whole point is that the accepting user isn't a member yet. accept_invitation() enforces the email-match binding and single-use/expiry rules; see its docstring."""

    token = request.data.get("token")
    if not token:
        return Response({"error": "Missing invitation token."}, status=400)

    try:
        membership = accept_invitation(token, request.user, request=request)
    except InvitationError as e:
        return Response({"error": str(e)}, status=400)

    return Response(_serialize_organization(membership.organization, membership))


# ============================================================
# Org-scoped audit log
# ============================================================

AUDIT_LOG_PAGE_SIZE = 50


@api_view(["GET"])
@permission_classes([HasOrgPermission("audit_logs.view")])
def organization_audit_logs_view(request, org_slug):
    """
    `?action=` filters to entries whose action exactly matches (the
    frontend gets the exact set of distinct actions to offer via
    `available_actions`, rather than guessing at free text); `?actor=`
    filters to one actor's username; `?page=` paginates
    AUDIT_LOG_PAGE_SIZE at a time. All three are optional - the
    previous hardcoded `[:200]` slice with no filtering at all is gone,
    since an organization with real history quickly exceeds 200 rows
    and had no way to see anything past it.
    """

    from django.core.paginator import Paginator

    logs = request.organization.audit_logs.select_related("actor").all()

    action = request.query_params.get("action", "").strip()
    if action:
        logs = logs.filter(action=action)

    actor = request.query_params.get("actor", "").strip()
    if actor:
        logs = logs.filter(actor__username=actor)

    try:
        page_number = int(request.query_params.get("page", 1))
    except (TypeError, ValueError):
        page_number = 1

    available_actions = list(
        request.organization.audit_logs.order_by("action").values_list("action", flat=True).distinct()
    )

    paginator = Paginator(logs, AUDIT_LOG_PAGE_SIZE)
    page = paginator.get_page(page_number)

    return Response({
        "audit_logs": [_serialize_audit_log(e, include_ip=True) for e in page.object_list],
        "page": page.number,
        "num_pages": paginator.num_pages,
        "count": paginator.count,
        "has_next": page.has_next(),
        "has_previous": page.has_previous(),
        "available_actions": available_actions,
    })


# ============================================================
# Platform Super Admin oversight (organizations.* - see
# org_permission_service's module docstring for why this is
# expressed as permissions on the existing platform Admin role
# rather than a second top-tier role)
# ============================================================

@api_view(["GET"])
@permission_classes([HasPagePermission("organizations.view_all")])
def platform_organizations_view(request):
    """
    Cross-organization listing - potentially every org on the platform,
    so this deliberately avoids the one-query-per-row pattern
    _serialize_organization() defaults to for a single org: owner
    labels and member counts are each computed in one batch query
    up front (_owner_labels_by_org_id / _member_counts_by_org_id) and
    threaded through per-row, rather than N+1 queries each.
    """

    organizations = list(Organization.objects.select_related("org_type").order_by("-created_at"))
    owner_labels = _owner_labels_by_org_id(organizations)
    member_counts = _member_counts_by_org_id(organizations)

    return Response({
        "organizations": [
            _serialize_organization(o, owner_label=owner_labels.get(o.id), member_count=member_counts.get(o.id, 0))
            for o in organizations
        ],
        "can_manage": user_has_permission(request.user, "organizations.manage"),
    })


@api_view(["POST"])
@permission_classes([HasPagePermission("organizations.manage")])
def platform_organization_action_view(request, org_slug):
    organization = get_object_or_404(Organization, slug=org_slug)
    action = request.data.get("action")

    if action == "suspend":
        suspend_organization(organization, request.user, request=request)
    elif action == "reactivate":
        reactivate_organization(organization, request.user, request=request)
    elif action == "delete":
        delete_organization(organization, request.user, request=request)
        return Response({"ok": True})
    else:
        return Response({"error": "Unknown action."}, status=400)

    return Response(_serialize_organization(organization))
