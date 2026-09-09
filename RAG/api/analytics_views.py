"""
Analytics endpoint for the React SPA - thin JSON wrapper around
RAG.views.analytics_view's exact same service calls
(stats_service.get_analytics_data/get_document_type_breakdown,
observability_service.get_performance_summary). No new aggregation
logic - every number here is computed by the same functions the
classic page already uses.

Company-wise + overall, same as Reports: that organization's own Owner
sees its Analytics (the same org-management surface rank Owner already
holds everywhere else - members, billing, settings - now includes
Analytics/Reports too), a platform Admin can additionally view
"overall" (Personal/platform) numbers or audit any SPECIFIC company's
Analytics via an explicit ?organization=<slug>|personal query param -
see org_permission_service.resolve_admin_scoped_organization()'s
docstring - and the Owner can also individually grant a Member
Analytics via the ordinary per-member Feature Access toggle (still
bounded by the company's Plan), same as every other FEATURE_CODES
entry; see user_can_view_org_analytics()'s docstring. A Member let in
this way sees only their own activity, never the whole company's - the
same Owner-vs-Member split dashboard_view already applies.
"""

from django.core.exceptions import PermissionDenied
from rest_framework.decorators import api_view, permission_classes

from rest_framework.response import Response

from ..services import observability_service
from ..services.org_member_feature_service import has_feature_access
from ..services.org_permission_service import ORG_ROLE_OWNER, get_user_org_role, resolve_admin_scoped_organization, user_can_view_org_analytics
from ..services.permission_service import user_has_permission
from ..services.stats_service import get_analytics_data, get_document_type_breakdown
from .permissions import HasPagePermission


@api_view(["GET"])
@permission_classes([HasPagePermission("pages.analytics")])
def analytics_view(request):
    organization, is_override = resolve_admin_scoped_organization(request)

    if not user_can_view_org_analytics(request.user, organization, feature_code="analytics"):
        raise PermissionDenied(
            "Only this organization's Owner, a Member granted Analytics access, or a platform Admin can view Analytics for it."
        )
    # Same reasoning as reports_views._check_report_access(): the org's
    # own Plan feature-gate additionally bounds what a MEMBER can do on
    # their own Plan (user_can_view_org_analytics already checked this
    # for the Member path above, but an Owner is bounded by the Plan
    # too - has_feature_access() applies that ceiling regardless of
    # role); it's skipped for an admin override auditing a company they
    # don't belong to, since that's already gated by is_admin() itself
    # via the override mechanism.
    if not is_override and not has_feature_access(request.user, organization, "analytics"):
        raise PermissionDenied(
            "This feature isn't included in your organization's plan, or has been restricted for your account."
        )

    # A Member sees only their own activity, never the rest of the
    # company's, even once granted access - matching dashboard_view's
    # existing Owner-vs-Member split. get_user_org_role() answers OWNER
    # for both a real Owner and an admin override (see its own
    # docstring), so this is False for both of those, same as
    # dashboard_view's scope_to_own.
    scope_to_own = organization is not None and get_user_org_role(request.user, organization) != ORG_ROLE_OWNER

    data = get_analytics_data(request.user, organization=organization, scope_to_own=scope_to_own)

    can_view_all = user_has_permission(request.user, "analytics.view_all")
    ai_performance = observability_service.get_performance_summary({
        **({} if can_view_all else {"user_id": request.user.id}),
        "organization": organization,
    })

    return Response({
        "data": data,
        "ai_performance": ai_performance,
        "document_types": get_document_type_breakdown(request.user, organization=organization, scope_to_own=scope_to_own),
        "can_view_ai_tasks": user_has_permission(request.user, "pages.ai_tasks"),
        "can_view_knowledge_base": user_has_permission(request.user, "pages.knowledge_base"),
    })
