"""
AI Tasks endpoints for the React SPA - thin JSON wrappers around the
exact same logic RAG/views.py's ai_task_create/ai_task_cancel/
ai_task_delete/ai_task_status/ai_task_results/ai_task_history already
run (same validation, same task_runner.submit() dispatch, same
owner-or-admin authorization on cancel/delete). export_csv stays a
plain Django view (not DRF) since it streams a CSV file, matching
RAG.views.ai_task_export/export_documents_report's own pattern.
"""

import csv
import json

from django.conf import settings
from django.core.exceptions import PermissionDenied
from django.core.paginator import Paginator
from django.http import HttpResponse, HttpResponseNotAllowed
from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response

from ..decorators import org_feature_required, permission_required
from ..models import AITaskRun, AITaskRunDocument
from ..services import billing_service
from ..services.activity_log_service import log_activity
from ..services.document_access_service import get_accessible_document_ids
from ..services.org_permission_service import resolve_request_organization
from ..services.permission_service import is_admin, is_super_admin
from ..services.reports_service import AI_TASK_RESULTS_HEADER, get_ai_task_result_rows
from .permissions import HasOrgFeatureAccess, HasPagePermission

AI_TASKS_NEEDING_REFERENCE = {AITaskRun.TaskType.VALIDATE}
AI_TASKS_ALLOWING_REFERENCE = {AITaskRun.TaskType.ANALYZE, AITaskRun.TaskType.VALIDATE}


def _serialize_run(run, **extra):
    return {
        "id": run.id,
        "task_type": run.task_type,
        "task_type_display": run.get_task_type_display(),
        "status": run.status,
        "status_display": run.get_status_display(),
        "cancel_requested": run.cancel_requested,
        "document_count": run.document_count,
        "error_message": run.error_message,
        "created_at": run.created_at,
        **extra,
    }


def _serialize_result(result):
    return {
        "id": result.id,
        "document_id": result.document_id,
        "rank": result.rank,
        "score": result.score,
        "title": result.title,
        "summary": result.summary,
        "data": result.data,
        "citations": result.citations,
    }


@api_view(["GET"])
@permission_classes([HasPagePermission("pages.ai_tasks"), HasOrgFeatureAccess("ai_tasks")])
def ai_tasks_config_view(request):
    return Response({
        "task_types": [{"value": value, "label": label} for value, label in AITaskRun.TaskType.choices],
        "max_documents": settings.AI_TASKS_MAX_DOCUMENTS,
        "tasks_needing_reference": list(AI_TASKS_NEEDING_REFERENCE),
        "tasks_allowing_reference": list(AI_TASKS_ALLOWING_REFERENCE),
    })


@api_view(["POST"])
@permission_classes([HasPagePermission("pages.ai_tasks"), HasOrgFeatureAccess("ai_tasks")])
def ai_task_create_view(request):
    task_type = request.data.get("task_type", "")

    if task_type not in AITaskRun.TaskType.values:
        return Response({"error": "Choose a task type."}, status=400)

    document_ids_raw = request.data.get("document_ids") or []
    reference_ids_raw = request.data.get("reference_document_ids") or []

    organization, _ = resolve_request_organization(request)
    accessible_ids = get_accessible_document_ids(request.user, organization=organization)

    def _valid_ids(raw_ids):
        parsed = []
        for value in raw_ids:
            try:
                doc_id = int(value)
            except (TypeError, ValueError):
                continue
            if doc_id in accessible_ids:
                parsed.append(doc_id)
        return parsed

    target_ids = _valid_ids(document_ids_raw)
    reference_ids = _valid_ids(reference_ids_raw) if task_type in AI_TASKS_ALLOWING_REFERENCE else []

    if len(target_ids) != len(document_ids_raw) or len(reference_ids) != len(reference_ids_raw):
        return Response({"error": "One or more selected documents are not available to you."}, status=400)

    if not target_ids:
        return Response({"error": "Select at least one document."}, status=400)

    if len(target_ids) > settings.AI_TASKS_MAX_DOCUMENTS:
        return Response({"error": f"You selected {len(target_ids)} documents; AI Task runs are limited to {settings.AI_TASKS_MAX_DOCUMENTS} documents per run."}, status=400)

    if task_type in AI_TASKS_NEEDING_REFERENCE and not reference_ids:
        return Response({"error": "This task requires at least one reference document."}, status=400)

    ai_credit_cost = len(target_ids) * billing_service.AI_CREDIT_COST_PER_AI_TASK_DOCUMENT

    try:
        billing_service.check_ai_task_limit_for(organization, request.user)
        billing_service.check_ai_credits_for(organization, request.user, ai_credit_cost)
    except billing_service.UsageLimitExceeded as e:
        return Response({"error": str(e), "code": "usage_limit_exceeded", "limit_type": e.limit_type}, status=402)

    config = request.data.get("config")
    if isinstance(config, str):
        try:
            config = json.loads(config or "{}")
        except (ValueError, TypeError):
            config = {}
    if not isinstance(config, dict):
        config = {}

    run = AITaskRun.objects.create(
        user=request.user, organization=organization, task_type=task_type, config=config, document_count=len(target_ids),
    )

    AITaskRunDocument.objects.bulk_create(
        [AITaskRunDocument(run=run, document_id=doc_id, role=AITaskRunDocument.Role.TARGET) for doc_id in target_ids]
        + [AITaskRunDocument(run=run, document_id=doc_id, role=AITaskRunDocument.Role.REFERENCE) for doc_id in reference_ids]
    )

    billing_service.deduct_ai_credits_for(
        organization, request.user, ai_credit_cost, f"AI Task run ({run.get_task_type_display()}, {len(target_ids)} document(s))", actor=request.user,
    )

    log_activity(
        actor=request.user,
        action="ai_task.created",
        description=f'{request.user.username} started an AI Task ({run.get_task_type_display()}) over {len(target_ids)} document(s)',
        request=request,
    )

    from ..services import task_runner
    from ..tasks import run_ai_task

    try:
        task_runner.submit(run_ai_task, run.id, key=run.id)
    except Exception:
        run.status = AITaskRun.Status.FAILED
        run.error_message = "Could not start this task due to an unexpected server error. Contact your administrator."
        run.save(update_fields=["status", "error_message"])

    return Response(_serialize_run(run), status=201)


@api_view(["GET"])
@permission_classes([HasPagePermission("pages.ai_tasks"), HasOrgFeatureAccess("ai_tasks")])
def ai_task_status_view(request, run_id):
    run = get_object_or_404(AITaskRun, id=run_id, user=request.user)

    return Response({
        "status": run.status,
        "cancel_requested": run.cancel_requested,
        "result_count": run.results.count(),
        "document_count": run.document_count,
        "error_message": run.error_message,
    })


@api_view(["POST"])
def ai_task_cancel_view(request, run_id):
    run = get_object_or_404(AITaskRun, id=run_id)

    if run.user_id != request.user.id and not (is_admin(request.user) or is_super_admin(request.user)):
        raise PermissionDenied("You don't have access to this run.")

    if run.status not in (AITaskRun.Status.PENDING, AITaskRun.Status.RUNNING):
        return Response({"error": "This run has already finished."}, status=400)

    run.cancel_requested = True
    run.save(update_fields=["cancel_requested"])

    from ..services import task_runner
    task_runner.cancel(run.id)

    return Response({"status": run.status, "cancel_requested": True})


@api_view(["POST"])
def ai_task_delete_view(request, run_id):
    run = get_object_or_404(AITaskRun, id=run_id)

    if run.user_id != request.user.id and not (is_admin(request.user) or is_super_admin(request.user)):
        raise PermissionDenied("You don't have access to this run.")

    if run.status in (AITaskRun.Status.PENDING, AITaskRun.Status.RUNNING):
        return Response({"error": "Cancel this run before deleting it."}, status=400)

    task_type_display = run.get_task_type_display()

    log_activity(
        actor=request.user,
        action="ai_task.deleted",
        description=f'{request.user.username} deleted an AI Task ({task_type_display})',
        request=request,
    )

    run.delete()

    return Response({"deleted": True})


@api_view(["GET"])
@permission_classes([HasPagePermission("pages.ai_tasks"), HasOrgFeatureAccess("ai_tasks")])
def ai_task_results_view(request, run_id):
    run = get_object_or_404(AITaskRun, id=run_id, user=request.user)

    per_document_results = list(run.results.filter(document__isnull=False).select_related("document"))
    corpus_results = list(run.results.filter(document__isnull=True))

    return Response({
        "run": _serialize_run(run),
        "per_document_results": [_serialize_result(r) for r in per_document_results],
        "corpus_results": [_serialize_result(r) for r in corpus_results],
        "result_count": len(per_document_results) + len(corpus_results),
    })


@api_view(["GET"])
@permission_classes([HasPagePermission("pages.ai_tasks"), HasOrgFeatureAccess("ai_tasks")])
def ai_task_history_view(request):
    organization, _ = resolve_request_organization(request)

    if organization is not None:
        # Same "tenant-shared, not per-uploader" scoping documents_list_view
        # uses for an active organization workspace - every active
        # member sees every run started inside that organization, not
        # just their own.
        runs = AITaskRun.objects.filter(organization=organization).order_by("-created_at")
    else:
        runs = AITaskRun.objects.filter(user=request.user, organization__isnull=True).order_by("-created_at")

    paginator = Paginator(runs, 15)
    page_obj = paginator.get_page(request.query_params.get("page"))

    return Response({
        "results": [_serialize_run(r) for r in page_obj],
        "page": page_obj.number,
        "num_pages": paginator.num_pages,
        "count": paginator.count,
        "has_previous": page_obj.has_previous(),
        "has_next": page_obj.has_next(),
    })


@permission_required("pages.ai_tasks")
@org_feature_required("ai_tasks")
def ai_task_export_view(request, run_id):
    """Plain Django view (not DRF) - streams a CSV file, same as RAG.views.ai_task_export."""

    if request.method != "GET":
        return HttpResponseNotAllowed(["GET"])

    run = get_object_or_404(AITaskRun, id=run_id, user=request.user)

    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = f'attachment; filename="ai_task_{run.id}_results.csv"'

    writer = csv.writer(response)
    writer.writerow(AI_TASK_RESULTS_HEADER)
    writer.writerows(get_ai_task_result_rows(run))

    return response
