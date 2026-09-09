"""
User Overview (/dashboard/) and Admin Overview (/admin/) as JSON -
reuses exactly the same service calls their Django-template
predecessors (RAG.views.user_dashboard / RAG.views.admin_dashboard_view)
already make. No aggregate query is reimplemented here.
"""

from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response

from ..models import FEATURE_CODES
from ..services.knowledge_service import get_knowledge_overview
from ..services.org_member_feature_service import has_feature_access
from ..services.org_permission_service import ORG_ROLE_OWNER, get_user_org_role, resolve_request_organization
from ..services.permission_service import get_user_access_snapshot
from ..services.stats_service import (
    get_dashboard_stats,
    get_document_type_breakdown,
    get_documents_over_time,
    get_kpi_trends,
    get_recent_activity,
    get_recent_documents_table,
    get_system_status,
)
from .permissions import HasAdminAreaAccess

# Mirrors RAG.views.DASHBOARD_CHART_RANGES - the "Documents Over Time"
# chart's `?range=` control on Admin Overview.
DASHBOARD_CHART_RANGES = (7, 14, 30)


@api_view(["GET"])
def dashboard_view(request):
    user = request.user
    organization, _ = resolve_request_organization(request)

    # A plain Member's Overview must show only their own activity, not
    # the whole company's - only the org Owner (or Personal Workspace,
    # organization is None) gets the whole-workspace numbers
    # _workspace_scope() otherwise always returns. See
    # stats_service._workspace_scope()'s scope_to_own docstring.
    scope_to_own = organization is not None and get_user_org_role(user, organization) != ORG_ROLE_OWNER

    try:
        chart_range = int(request.query_params.get("range", 7))
    except (TypeError, ValueError):
        chart_range = 7
    if chart_range not in DASHBOARD_CHART_RANGES:
        chart_range = 7

    stats = get_dashboard_stats(user, organization=organization, scope_to_own=scope_to_own)
    activity = get_recent_activity(user, organization=organization, scope_to_own=scope_to_own)
    knowledge_overview = get_knowledge_overview(user, organization=organization)
    document_types = get_document_type_breakdown(user, organization=organization, scope_to_own=scope_to_own)
    documents_over_time = get_documents_over_time(user, days=chart_range, organization=organization, scope_to_own=scope_to_own)
    recent_documents_table = get_recent_documents_table(user, organization=organization, scope_to_own=scope_to_own)
    # Same "same design, but the org's Plan can hide things" gate
    # analytics_views.py/reports_views.py already apply at the page
    # level (HasOrgFeatureAccess) - here it's per-section instead of
    # per-page, so User Overview/Company Owner Overview can hide just
    # the Knowledge Base or AI Tasks card rather than the whole page.
    # Works for both Personal (organization=None, personal Plan
    # ceiling) and Company (org Plan ceiling + per-member override) -
    # see org_member_feature_service.has_feature_access()'s docstring.
    feature_access = {code: has_feature_access(user, organization, code) for code in FEATURE_CODES}
    role, can_view_admin_area, user_permissions = get_user_access_snapshot(user)

    # Same merged-and-sorted shape context_processors.sidebar_status
    # builds for the sidebar/topbar - reusing the recent_documents/
    # recent_questions already fetched above instead of re-querying,
    # since user_dashboard.html's Activity Feed card renders that same
    # global context var.
    #
    # PRIVACY: when `organization` is set and this viewer is the Owner
    # (scope_to_own=False above), _workspace_scope() deliberately
    # returns the WHOLE organization's QueryLog rows for aggregate
    # stats purposes - but that means `activity["recent_questions"]`
    # can contain OTHER members' rows too. Question text is content,
    # not metadata (same boundary RAG.services.queries_service /
    # admin_queries_views.py enforce for Admin > Queries -
    # "queries.view_content", never granted merely by being an
    # Owner/Admin) - so only ever show the real text for a row that
    # belongs to the viewer themselves; every other member's row still
    # shows up (an Owner can see THAT someone asked something) with a
    # generic label instead of what they actually asked.
    events = [
        {"icon": "file-arrow-up", "text": f'"{doc.title}" uploaded', "at": doc.uploaded_at}
        for doc in activity["recent_documents"]
    ] + [
        {
            "icon": "chat-circle",
            "text": f'Asked: "{log.question[:60]}"' if log.user_id == user.id else "Asked a question",
            "at": log.created_at,
        }
        for log in activity["recent_questions"]
    ]
    events.sort(key=lambda e: e["at"], reverse=True)

    return Response({
        "stats": {
            "total_documents": stats["total_documents"],
            "total_chunks": stats["total_chunks"],
            "questions_asked": stats["questions_asked"],
            "today_queries": stats["today_queries"],
            "avg_response_time": stats["avg_response_time"],
            "storage_used": stats["storage_used"],
            "ai_task_runs": stats["ai_task_runs"],
        },
        "kpi_trends": get_kpi_trends(user, organization=organization, scope_to_own=scope_to_own),
        "documents_over_time": documents_over_time,
        "documents_over_time_range": chart_range,
        "documents_over_time_ranges": list(DASHBOARD_CHART_RANGES),
        "document_types": document_types,
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
            for row in recent_documents_table
        ],
        "feature_access": feature_access,
        "knowledge_overview": {
            "total_entities": knowledge_overview.get("total_entities", 0),
            "total_relationships": knowledge_overview.get("total_relationships", 0),
            "total_sources": knowledge_overview.get("total_sources", 0),
        },
        "recent_documents": [
            {
                "id": doc.id,
                "title": doc.title,
                "file_type": doc.file_type,
                "chunk_count": doc.chunk_count,
                "uploaded_at": doc.uploaded_at.isoformat(),
            }
            for doc in activity["recent_documents"]
        ],
        "recent_questions": [
            {
                "id": log.id,
                "question": log.question if log.user_id == user.id else None,
                "confidence": log.confidence,
                "created_at": log.created_at.isoformat(),
            }
            for log in activity["recent_questions"]
        ],
        "recent_ai_task_runs": [
            {
                "id": run.id,
                "task_type_display": run.get_task_type_display(),
                "status_display": run.get_status_display(),
                "created_at": run.created_at.isoformat(),
            }
            for run in activity["recent_ai_task_runs"]
        ],
        "activity_feed": [
            {"icon": e["icon"], "text": e["text"], "at": e["at"].isoformat()}
            for e in events[:5]
        ],
        "permissions": user_permissions,
        "can_view_admin_area": can_view_admin_area,
    })


@api_view(["GET"])
@permission_classes([HasAdminAreaAccess])
def admin_overview_view(request):
    """
    Admin Overview (/admin/) as JSON - KPI cards with trend
    sparklines, the Documents Over Time / Document Types / Topics by
    Category charts, System Status, and the Recent Documents table.
    Gated the same coarse way as its template predecessor
    (HasAdminAreaAccess, see RAG.decorators.admin_area_required) - not
    behind any individual permission - since every role that gets the
    admin sidebar shell must always have an Overview to land on.
    """

    user = request.user
    organization, _ = resolve_request_organization(request)

    try:
        chart_range = int(request.query_params.get("range", 7))
    except (TypeError, ValueError):
        chart_range = 7
    if chart_range not in DASHBOARD_CHART_RANGES:
        chart_range = 7

    stats = get_dashboard_stats(user, organization=organization)
    activity = get_recent_activity(user, organization=organization)
    knowledge_overview = get_knowledge_overview(user, organization=organization)
    document_types = get_document_type_breakdown(user, organization=organization)
    documents_over_time = get_documents_over_time(user, days=chart_range, organization=organization)
    recent_documents_table = get_recent_documents_table(user, organization=organization)
    role, can_view_admin_area, user_permissions = get_user_access_snapshot(user)

    # PRIVACY: see dashboard_view's identical comment above -
    # activity["recent_questions"] can span every member of whichever
    # organization is currently resolved, and question text is content
    # that requires "queries.view_content" (Admin > Queries' own
    # boundary), never granted just by reaching the admin area. Only
    # ever show the real text for the viewer's own row.
    events = [
        {"icon": "file-arrow-up", "text": f'"{doc.title}" uploaded', "at": doc.uploaded_at}
        for doc in activity["recent_documents"]
    ] + [
        {
            "icon": "chat-circle",
            "text": f'Asked: "{log.question[:60]}"' if log.user_id == user.id else "Asked a question",
            "at": log.created_at,
        }
        for log in activity["recent_questions"]
    ]
    events.sort(key=lambda e: e["at"], reverse=True)

    return Response({
        "stats": {
            "total_documents": stats["total_documents"],
            "total_chunks": stats["total_chunks"],
            "today_queries": stats["today_queries"],
            "storage_used": stats["storage_used"],
            "ai_task_runs": stats["ai_task_runs"],
        },
        "kpi_trends": get_kpi_trends(user, organization=organization),
        "documents_over_time": documents_over_time,
        "documents_over_time_range": chart_range,
        "documents_over_time_ranges": list(DASHBOARD_CHART_RANGES),
        "document_types": document_types,
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
            for row in recent_documents_table
        ],
        "knowledge_overview": {
            "total_entities": knowledge_overview.get("total_entities", 0),
            "total_relationships": knowledge_overview.get("total_relationships", 0),
            "total_sources": knowledge_overview.get("total_sources", 0),
            "category_breakdown": knowledge_overview.get("category_breakdown", []),
        },
        "recent_documents": [
            {
                "id": doc.id,
                "title": doc.title,
                "uploaded_at": doc.uploaded_at.isoformat(),
                "file_size": doc.file_size,
            }
            for doc in activity["recent_documents"]
        ],
        "recent_ai_task_runs": [
            {
                "id": run.id,
                "task_type_display": run.get_task_type_display(),
                "status": run.status,
                "status_display": run.get_status_display(),
                "document_count": run.document_count,
                "created_at": run.created_at.isoformat(),
            }
            for run in activity["recent_ai_task_runs"]
        ],
        "activity_feed": [
            {"icon": e["icon"], "text": e["text"], "at": e["at"].isoformat()}
            for e in events[:5]
        ],
        "system_status": get_system_status(),
        "permissions": user_permissions,
        "can_view_admin_area": can_view_admin_area,
    })
