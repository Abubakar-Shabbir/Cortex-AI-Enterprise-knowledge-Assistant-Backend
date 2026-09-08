"""
Platform-admin, whole-system overview - the per-company + system-wide
KPI/AI-usage dashboard behind /admin/system-overview (frontend
AdminSystemOverview.jsx). Distinct from dashboard_views.admin_overview_view
(workspace-scoped: personal or one company at a time) - this always
spans every organization and every Personal Workspace on the platform
at once, so it's gated on "organizations.view_all" (the same
permission /admin/companies already requires) rather than the coarser
HasAdminAreaAccess, since it surfaces more cross-company detail than
the generic Admin Overview does.
"""

from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response

from ..models import Organization
from ..services import billing_service, observability_service
from ..services.stats_service import get_system_wide_stats
from .organizations_views import _member_counts_by_org_id, _owner_labels_by_org_id
from .permissions import HasPagePermission


@api_view(["GET"])
@permission_classes([HasPagePermission("organizations.view_all")])
def admin_system_overview_view(request):
    organizations = list(Organization.objects.select_related("subscription__plan", "org_type").order_by("-created_at"))
    owner_labels = _owner_labels_by_org_id(organizations)
    member_counts = _member_counts_by_org_id(organizations)

    org_rows = []
    for org in organizations:
        usage = billing_service.get_organization_usage(org)
        plan = usage.get("plan")
        org_rows.append({
            "slug": org.slug,
            "name": org.name,
            "status": org.status,
            "owner": owner_labels.get(org.id),
            "member_count": member_counts.get(org.id, 0),
            "document_count": org.documents.count(),
            "queries_used": usage.get("queries_used"),
            "ai_task_runs_used": usage.get("ai_task_runs_used"),
            "storage_used_bytes": usage.get("storage_used_bytes"),
            "plan_name": plan["name"] if plan else None,
            "included_features": plan["included_features"] if plan else None,
            "unlimited": usage.get("unlimited", True),
        })

    return Response({
        "system_stats": get_system_wide_stats(),
        "organizations": org_rows,
        "ai_performance": observability_service.get_performance_summary({}),
    })
