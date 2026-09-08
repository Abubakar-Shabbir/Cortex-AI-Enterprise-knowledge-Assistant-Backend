"""
Reports endpoints for the React SPA - thin JSON wrapper around
RAG.views.reports_view's exact same service calls, plus one plain
Django (non-DRF) view per CSV export, mirroring
RAG.views.export_documents_report/export_usage_report/
export_comparison_report/export_ai_task_runs_report/
export_knowledge_topics_report and ai_tasks_views.ai_task_export_view's
own streamed-CSV pattern. No new aggregation/row-building logic.
"""

import csv

from django.db.models import Sum
from django.http import HttpResponse, HttpResponseNotAllowed
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response

from ..decorators import org_analytics_admin_required, org_feature_required, permission_required
from ..models import AITaskRun, Document, QueryLog
from ..services.knowledge_service import get_knowledge_overview
from ..services.org_permission_service import resolve_request_organization
from ..services.permission_service import user_has_permission
from ..services.reports_service import (
    AI_TASK_RUNS_REPORT_HEADER,
    COMPARISON_REPORT_HEADER,
    DOCUMENTS_REPORT_HEADER,
    KNOWLEDGE_TOPICS_REPORT_HEADER,
    USAGE_REPORT_HEADER,
    get_ai_task_runs_report_rows,
    get_comparison_report_rows,
    get_documents_report_rows,
    get_knowledge_topics_report_rows,
    get_usage_report_rows,
)
from ..services.stats_service import get_comparison_report_data, get_document_type_breakdown, get_kpi_trends
from ..utils.formatting import format_bytes
from .permissions import HasOrgFeatureAccess, HasPagePermission, RestrictsOrgAnalyticsToAdmin


@api_view(["GET"])
@permission_classes([HasPagePermission("pages.reports"), HasOrgFeatureAccess("reports"), RestrictsOrgAnalyticsToAdmin])
def reports_view(request):
    organization, _ = resolve_request_organization(request)

    documents = (
        Document.objects.filter(organization=organization)
        if organization is not None
        else Document.objects.filter(user=request.user, organization__isnull=True)
    )
    ai_task_runs = (
        AITaskRun.objects.filter(organization=organization)
        if organization is not None
        else AITaskRun.objects.filter(user=request.user, organization__isnull=True)
    )

    return Response({
        "document_count": documents.count(),
        "total_storage": format_bytes(documents.aggregate(total=Sum("file_size"))["total"] or 0),
        "question_count": QueryLog.objects.filter(user=request.user, organization=organization).count(),
        "ai_task_run_count": ai_task_runs.count(),
        "topic_count": get_knowledge_overview(request.user, organization=organization)["total_entities"],
        # stats_service.py now threads `organization` through every
        # aggregate it exposes (see its _workspace_scope() helper), so
        # these reflect the active organization workspace exactly like
        # the fields above, instead of always showing Personal
        # Workspace trend data regardless of which workspace is active.
        "comparison": get_comparison_report_data(request.user, organization=organization),
        "kpi_trends": get_kpi_trends(request.user, organization=organization),
        "document_types": get_document_type_breakdown(request.user, organization=organization),
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
@org_feature_required("reports")
@org_analytics_admin_required
def export_documents_report_view(request):
    if request.method != "GET":
        return HttpResponseNotAllowed(["GET"])
    organization, _ = resolve_request_organization(request)
    return _csv_response("documents_report.csv", DOCUMENTS_REPORT_HEADER, get_documents_report_rows(request.user, organization=organization))


@permission_required("pages.reports")
@org_feature_required("reports")
@org_analytics_admin_required
def export_usage_report_view(request):
    if request.method != "GET":
        return HttpResponseNotAllowed(["GET"])
    organization, _ = resolve_request_organization(request)
    return _csv_response("usage_report.csv", USAGE_REPORT_HEADER, get_usage_report_rows(request.user, organization=organization))


@permission_required("pages.reports")
@org_feature_required("reports")
@org_analytics_admin_required
def export_comparison_report_view(request):
    if request.method != "GET":
        return HttpResponseNotAllowed(["GET"])
    organization, _ = resolve_request_organization(request)
    return _csv_response(
        "comparison_report.csv", COMPARISON_REPORT_HEADER,
        get_comparison_report_rows(get_comparison_report_data(request.user, organization=organization)),
    )


@permission_required("pages.reports")
@org_feature_required("reports")
@org_analytics_admin_required
def export_ai_task_runs_report_view(request):
    if request.method != "GET":
        return HttpResponseNotAllowed(["GET"])
    organization, _ = resolve_request_organization(request)
    return _csv_response("ai_task_runs_report.csv", AI_TASK_RUNS_REPORT_HEADER, get_ai_task_runs_report_rows(request.user, organization=organization))


@permission_required("pages.reports")
@org_feature_required("reports")
@org_analytics_admin_required
def export_knowledge_topics_report_view(request):
    if request.method != "GET":
        return HttpResponseNotAllowed(["GET"])
    organization, _ = resolve_request_organization(request)
    return _csv_response("knowledge_topics_report.csv", KNOWLEDGE_TOPICS_REPORT_HEADER, get_knowledge_topics_report_rows(request.user, organization=organization))
