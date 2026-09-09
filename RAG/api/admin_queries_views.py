"""
Admin > Queries endpoints for the React SPA - thin JSON wrappers
around queries_service.py's filter/sort/analytics. Deliberately
metadata-only, full stop: no endpoint here ever returns or accepts a
search against `question`/`answer`/`sources` text, and there is no
content-detail view to view/audit-log in the first place - an admin
with "queries.view_all_logs" sees that a query happened (owner,
status, search method, confidence, response time, source count,
flagged state, timestamp), never what was actually asked or answered.
(Django's own /django-admin/ QueryLog page and the classic
RAG.views.admin_query_detail_view are a separate surface that still
gates real content behind "queries.view_content" for explicitly
authorized auditing - this module just never offers that capability at
all, by design.)
"""

import csv

from django.core.paginator import Paginator
from django.http import HttpResponse, HttpResponseNotAllowed
from rest_framework.decorators import api_view, permission_classes

from rest_framework.response import Response

from ..decorators import permission_required
from ..models import QueryLog
from ..services.queries_service import annotate_status, filter_and_sort_queries, get_queries_analytics, get_search_methods
from ..services.reports_service import QUERIES_REPORT_HEADER_METADATA, get_queries_report_rows
from .permissions import HasPagePermission


def _serialize_log(log):
    return {
        "id": log.id,
        "owner": log.user.username,
        "status_answered": log.status_answered,
        "status_label": log.status_label,
        "search_method": log.search_method,
        "confidence": log.confidence,
        "response_time_ms": log.response_time_ms,
        "source_count": len(log.sources or []),
        "created_at": log.created_at,
        "is_flagged": log.is_flagged,
    }


@api_view(["GET"])
@permission_classes([HasPagePermission("queries.view_all_logs")])
def admin_queries_view(request):
    logs = filter_and_sort_queries(request.query_params, request_user=request.user)
    analytics = get_queries_analytics(logs)

    paginator = Paginator(logs, 20)
    page_obj = paginator.get_page(request.query_params.get("page"))
    annotate_status(page_obj.object_list)

    return Response({
        "results": [_serialize_log(log) for log in page_obj],
        "page": page_obj.number,
        "num_pages": paginator.num_pages,
        "count": paginator.count,
        "has_previous": page_obj.has_previous(),
        "has_next": page_obj.has_next(),
        "analytics": analytics,
        "search_methods": get_search_methods(),
    })


@api_view(["POST"])
@permission_classes([HasPagePermission("queries.view_all_logs")])
def admin_query_toggle_flag_view(request, log_id):
    log = QueryLog.objects.filter(id=log_id).first()
    if log is None:
        return Response({"error": "Not found."}, status=404)
    log.is_flagged = not log.is_flagged
    log.save(update_fields=["is_flagged"])
    return Response({"id": log.id, "is_flagged": log.is_flagged})


@permission_required("queries.view_all_logs")
def export_queries_report_view(request):
    if request.method != "GET":
        return HttpResponseNotAllowed(["GET"])

    logs = filter_and_sort_queries(request.GET, request_user=request.user).select_related("user")

    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = 'attachment; filename="queries_report.csv"'
    writer = csv.writer(response)
    writer.writerow(QUERIES_REPORT_HEADER_METADATA)
    writer.writerows(get_queries_report_rows(logs, include_content=False))
    return response
