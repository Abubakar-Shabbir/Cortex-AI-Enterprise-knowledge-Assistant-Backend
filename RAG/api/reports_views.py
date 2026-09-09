"""
Reports endpoints for the React SPA - thin JSON wrapper around
RAG.views.reports_view's exact same service calls, plus one plain
Django (non-DRF) view per CSV export, mirroring
RAG.views.export_documents_report/export_usage_report/
export_comparison_report/export_ai_task_runs_report/
export_knowledge_topics_report and ai_tasks_views.ai_task_export_view's
own streamed-CSV pattern. No new aggregation/row-building logic.

Company-wise + overall: every view below resolves its workspace via
org_permission_service.resolve_admin_scoped_organization() rather than
calling resolve_request_organization() (the X-Organization-Slug
header) directly - that organization's own Owner sees its Reports (the
same org-management surface rank Owner already holds everywhere else -
members, billing, settings), a platform Admin can additionally pass an
explicit `?organization=<slug>` (or `?organization=personal`) to pull
ANY company's report, or their own overall/Personal Workspace report,
without first switching their active workspace to it (which they
usually can't do at all - an Admin is rarely a real member of the
company they need to audit), and the Owner can also individually grant
a Member Reports via the ordinary per-member Feature Access toggle
(still bounded by the company's Plan) - see
_check_report_access()/user_can_view_org_analytics()'s docstrings. A
Member let in this way sees only their own activity/content, never the
whole company's - see _report_scope()/get_usage_report_rows()'s
docstrings.
"""

import csv

from django.core.exceptions import PermissionDenied
from django.db.models import Sum
from django.http import HttpResponse, HttpResponseNotAllowed
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response

from ..decorators import permission_required
from ..models import AITaskRun, Document, QueryLog
from ..services.knowledge_service import get_knowledge_overview
from ..services.org_member_feature_service import has_feature_access
from ..services.org_permission_service import (
    ORG_ROLE_OWNER, get_user_org_role, resolve_admin_scoped_organization, user_can_view_org_analytics,
)
from ..services.permission_service import user_has_permission
from ..services.reports_service import (
    AI_TASK_RUNS_REPORT_HEADER,
    COMPARISON_REPORT_HEADER,
    DOCUMENTS_REPORT_HEADER,
    KNOWLEDGE_TOPICS_REPORT_HEADER,
    USAGE_REPORT_HEADER,
    USAGE_REPORT_HEADER_WORKSPACE,
    get_ai_task_runs_report_rows,
    get_comparison_report_rows,
    get_documents_report_rows,
    get_knowledge_topics_report_rows,
    get_usage_report_rows,
)
from ..services.stats_service import get_comparison_report_data, get_document_type_breakdown, get_kpi_trends
from ..utils.formatting import format_bytes
from .permissions import HasPagePermission


def _check_report_access(request, organization, is_override):
    """
    Only this organization's own Owner, a Member individually granted
    Reports via their own Plan-bounded feature toggle, or a platform
    Admin may proceed (user_can_view_org_analytics) - a plain,
    ungranted Member holds none of the org-management surface. The
    org's own Plan feature-gate (`has_feature_access`) is additionally
    checked here too - redundant for the Member path (already implied
    by user_can_view_org_analytics's own check) but still load-bearing
    for the Owner, who is bounded by the Plan just not by a per-member
    override - and is skipped for an admin override auditing a company
    they don't belong to, since that's already gated by is_admin()
    itself via the override mechanism.
    """

    if not user_can_view_org_analytics(request.user, organization, feature_code="reports"):
        raise PermissionDenied(
            "Only this organization's Owner, a Member granted Reports access, or a platform Admin can view Reports for it."
        )
    if not is_override and not has_feature_access(request.user, organization, "reports"):
        raise PermissionDenied(
            "This feature isn't included in your organization's plan, or has been restricted for your account."
        )


def _report_scope(request, organization):
    """
    True when the viewer should see only their own activity, never the
    rest of the company's - a Member let in via their own Feature
    Access grant rather than the Owner rank. get_user_org_role()
    answers OWNER for both a real Owner and a platform Admin's
    oversight override (see its own docstring), so this is False for
    both of those, same as dashboard_view's existing scope_to_own.
    """

    return organization is not None and get_user_org_role(request.user, organization) != ORG_ROLE_OWNER


@api_view(["GET"])
@permission_classes([HasPagePermission("pages.reports")])
def reports_view(request):
    organization, is_override = resolve_admin_scoped_organization(request)
    _check_report_access(request, organization, is_override)
    scope_to_own = _report_scope(request, organization)

    documents = (
        Document.objects.filter(organization=organization, **({"user": request.user} if scope_to_own else {}))
        if organization is not None
        else Document.objects.filter(user=request.user, organization__isnull=True)
    )
    ai_task_runs = (
        AITaskRun.objects.filter(organization=organization, **({"user": request.user} if scope_to_own else {}))
        if organization is not None
        else AITaskRun.objects.filter(user=request.user, organization__isnull=True)
    )
    # Whole-company count once inside an organization workspace for the
    # Owner/admin-override case - same as Documents/AI Task Runs above,
    # and matching what export_usage_report_view actually exports (see
    # get_usage_report_rows()'s whole_workspace docstring); narrowed to
    # just the viewer's own rows for a Member let in via their own
    # Feature Access grant instead.
    questions = (
        QueryLog.objects.filter(organization=organization, **({"user": request.user} if scope_to_own else {}))
        if organization is not None
        else QueryLog.objects.filter(user=request.user, organization__isnull=True)
    )

    return Response({
        "document_count": documents.count(),
        "total_storage": format_bytes(documents.aggregate(total=Sum("file_size"))["total"] or 0),
        "question_count": questions.count(),
        "ai_task_run_count": ai_task_runs.count(),
        # Topics are always accessible-document-scoped per user
        # regardless of role (get_knowledge_overview() already only
        # ever counts documents `request.user` can see), so this needs
        # no separate scope_to_own handling.
        "topic_count": get_knowledge_overview(request.user, organization=organization)["total_entities"],
        # stats_service.py now threads `organization` through every
        # aggregate it exposes (see its _workspace_scope() helper), so
        # these reflect the active organization workspace exactly like
        # the fields above, instead of always showing Personal
        # Workspace trend data regardless of which workspace is active.
        "comparison": get_comparison_report_data(request.user, organization=organization, scope_to_own=scope_to_own),
        "kpi_trends": get_kpi_trends(request.user, organization=organization, scope_to_own=scope_to_own),
        "document_types": get_document_type_breakdown(request.user, organization=organization, scope_to_own=scope_to_own),
        "can_view_ai_tasks": user_has_permission(request.user, "pages.ai_tasks"),
        "can_view_knowledge_base": user_has_permission(request.user, "pages.knowledge_base"),
    })


def _csv_response(filename, header, rows):
    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = f'attachment; filename="{filename}"'
    writer = csv.writer(response)
    writer.writerow(header)
    writer.writerows(rows)
    return response


@permission_required("pages.reports")
def export_documents_report_view(request):
    if request.method != "GET":
        return HttpResponseNotAllowed(["GET"])
    organization, is_override = resolve_admin_scoped_organization(request)
    _check_report_access(request, organization, is_override)
    scope_to_own = _report_scope(request, organization)
    return _csv_response(
        "documents_report.csv", DOCUMENTS_REPORT_HEADER,
        get_documents_report_rows(request.user, organization=organization, scope_to_own=scope_to_own),
    )


@permission_required("pages.reports")
def export_usage_report_view(request):
    if request.method != "GET":
        return HttpResponseNotAllowed(["GET"])
    organization, is_override = resolve_admin_scoped_organization(request)
    _check_report_access(request, organization, is_override)
    # Whole-team, content-free rows for the Owner/admin-override case;
    # a Member let in via their own Feature Access grant instead gets
    # their own rows WITH content, same as Personal Workspace's own-
    # content model - see get_usage_report_rows()'s docstring.
    whole_workspace = organization is not None and not _report_scope(request, organization)
    header = USAGE_REPORT_HEADER_WORKSPACE if whole_workspace else USAGE_REPORT_HEADER
    rows = get_usage_report_rows(request.user, organization=organization, whole_workspace=whole_workspace)
    return _csv_response("usage_report.csv", header, rows)


@permission_required("pages.reports")
def export_comparison_report_view(request):
    if request.method != "GET":
        return HttpResponseNotAllowed(["GET"])
    organization, is_override = resolve_admin_scoped_organization(request)
    _check_report_access(request, organization, is_override)
    scope_to_own = _report_scope(request, organization)
    return _csv_response(
        "comparison_report.csv", COMPARISON_REPORT_HEADER,
        get_comparison_report_rows(get_comparison_report_data(request.user, organization=organization, scope_to_own=scope_to_own)),
    )


@permission_required("pages.reports")
def export_ai_task_runs_report_view(request):
    if request.method != "GET":
        return HttpResponseNotAllowed(["GET"])
    organization, is_override = resolve_admin_scoped_organization(request)
    _check_report_access(request, organization, is_override)
    scope_to_own = _report_scope(request, organization)
    return _csv_response(
        "ai_task_runs_report.csv", AI_TASK_RUNS_REPORT_HEADER,
        get_ai_task_runs_report_rows(request.user, organization=organization, scope_to_own=scope_to_own),
    )


@permission_required("pages.reports")
def export_knowledge_topics_report_view(request):
    if request.method != "GET":
        return HttpResponseNotAllowed(["GET"])
    organization, is_override = resolve_admin_scoped_organization(request)
    _check_report_access(request, organization, is_override)
    return _csv_response("knowledge_topics_report.csv", KNOWLEDGE_TOPICS_REPORT_HEADER, get_knowledge_topics_report_rows(request.user, organization=organization))
