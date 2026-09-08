"""
Billing endpoints for the React SPA - internal bookkeeping only, no
real payment processor. Plan CRUD is Super-Admin-only (gated on the
platform "billing.manage_plans"/"billing.view_all" permissions, never
anything org-scoped) - the structural guarantee behind "an Owner can
never set a limit above what their assigned Plan grants": there is no
Owner-reachable code path in this file that writes to Plan at all.
Direct plan ASSIGNMENT (assign_plan_view) is also Super-Admin-only and
instant - the separate, Owner-initiated REQUEST path (organizations_
views.organization_plan_request_view) needs admin approval via
plan_requests_view/plan_request_action_view below before it ever
touches a Subscription. organization_billing_view is the one read-only
exception, for an org's own Owner. See personal_billing_views.py for
the Personal-Workspace counterpart of everything here.
"""

from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response

from ..models import FEATURE_CODES, FEATURE_LABELS, Organization, Plan, PlanChangeRequest
from ..services import billing_service
from ..services.billing_service import BillingServiceError
from .permissions import HasOrgPermission, HasPagePermission


def _serialize_plan(plan):
    return {
        "id": plan.id,
        "slug": plan.slug,
        "name": plan.name,
        "description": plan.description,
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
        "is_active": plan.is_active,
    }


def _serialize_plan_request(pr):
    return {
        "id": pr.id,
        "organization": {"slug": pr.organization.slug, "name": pr.organization.name} if pr.organization_id else None,
        "user": {"id": pr.user_id, "username": pr.user.username} if pr.user_id else None,
        "requested_plan": _serialize_plan(pr.requested_plan),
        "status": pr.status,
        "requested_by": pr.requested_by.username if pr.requested_by_id else None,
        "reviewed_by": pr.reviewed_by.username if pr.reviewed_by_id else None,
        "reviewed_at": pr.reviewed_at.isoformat() if pr.reviewed_at else None,
        "admin_note": pr.admin_note,
        "created_at": pr.created_at.isoformat(),
    }


_PLAN_FIELDS = (
    "name", "description", "is_active", "plan_type", "price", "currency", "billing_interval",
    "max_queries_per_month", "max_ai_task_runs_per_month", "max_storage_bytes", "max_seats",
    "included_credits", "allow_credit_purchase", "included_features",
)


@api_view(["GET", "POST"])
@permission_classes([HasPagePermission("billing.manage_plans")])
def plans_view(request):
    if request.method == "GET":
        plans = Plan.objects.all()
        plan_type = request.query_params.get("plan_type")
        if plan_type:
            plans = plans.filter(plan_type=plan_type)
        return Response({
            "plans": [_serialize_plan(p) for p in plans],
            "feature_catalog": [{"code": code, "label": FEATURE_LABELS[code]} for code in FEATURE_CODES],
        })

    try:
        plan = billing_service.create_plan(
            **{k: request.data[k] for k in _PLAN_FIELDS if k in request.data and k != "is_active"},
            actor=request.user, request=request,
        )
    except BillingServiceError as e:
        return Response({"error": str(e)}, status=400)
    return Response(_serialize_plan(plan), status=201)


@api_view(["GET", "PATCH"])
@permission_classes([HasPagePermission("billing.manage_plans")])
def plan_detail_view(request, plan_id):
    plan = get_object_or_404(Plan, id=plan_id)

    if request.method == "GET":
        return Response(_serialize_plan(plan))

    fields = {k: request.data[k] for k in _PLAN_FIELDS if k in request.data}
    try:
        billing_service.update_plan(plan, actor=request.user, request=request, **fields)
    except BillingServiceError as e:
        return Response({"error": str(e)}, status=400)
    return Response(_serialize_plan(plan))


@api_view(["GET"])
@permission_classes([HasPagePermission("billing.view_all")])
def platform_organizations_billing_view(request):
    organizations = Organization.objects.select_related("subscription__plan").order_by("name")
    return Response({
        "organizations": [
            {
                "slug": org.slug,
                "name": org.name,
                "usage": billing_service.get_organization_usage(org),
            }
            for org in organizations
        ],
    })


@api_view(["POST", "DELETE"])
@permission_classes([HasPagePermission("billing.manage_plans")])
def assign_plan_view(request, org_slug):
    organization = get_object_or_404(Organization, slug=org_slug)

    if request.method == "DELETE":
        billing_service.unassign_plan(organization, request.user, request=request)
        return Response({"ok": True})

    plan = get_object_or_404(Plan, id=request.data.get("plan_id"))
    billing_service.assign_plan(organization, plan, request.user, request=request)
    return Response(billing_service.get_organization_usage(organization))


@api_view(["GET"])
@permission_classes([HasOrgPermission("billing.view")])
def organization_billing_view(request, org_slug):
    usage = billing_service.get_organization_usage(request.organization)
    available_plans = Plan.objects.filter(plan_type=Plan.PlanType.COMPANY, is_active=True)
    pending_request = PlanChangeRequest.objects.filter(
        organization=request.organization, status=PlanChangeRequest.Status.PENDING,
    ).select_related("requested_plan").first()

    return Response({
        **usage,
        "available_plans": [_serialize_plan(p) for p in available_plans],
        "pending_request": _serialize_plan_request(pending_request) if pending_request else None,
    })


@api_view(["POST"])
@permission_classes([HasOrgPermission("billing.view")])
def organization_plan_request_view(request, org_slug):
    plan = get_object_or_404(Plan, id=request.data.get("plan_id"), plan_type=Plan.PlanType.COMPANY)
    try:
        plan_request = billing_service.request_plan_change(
            plan, request.user, organization=request.organization, request=request,
        )
    except BillingServiceError as e:
        return Response({"error": str(e)}, status=400)
    return Response(_serialize_plan_request(plan_request), status=201)


@api_view(["GET"])
@permission_classes([HasPagePermission("billing.manage_plans")])
def plan_requests_view(request):
    requests_qs = PlanChangeRequest.objects.select_related("organization", "user", "requested_plan", "requested_by", "reviewed_by")
    status_filter = request.query_params.get("status")
    if status_filter:
        requests_qs = requests_qs.filter(status=status_filter)
    return Response({"plan_requests": [_serialize_plan_request(pr) for pr in requests_qs]})


@api_view(["POST"])
@permission_classes([HasPagePermission("billing.manage_plans")])
def plan_request_action_view(request, request_id):
    plan_request = get_object_or_404(PlanChangeRequest, id=request_id)
    action = request.data.get("action")

    try:
        if action == "approve":
            billing_service.approve_plan_request(plan_request, request.user, request=request)
        elif action == "reject":
            billing_service.reject_plan_request(plan_request, request.user, note=request.data.get("note", ""), request=request)
        else:
            return Response({"error": "Unknown action."}, status=400)
    except BillingServiceError as e:
        return Response({"error": str(e)}, status=400)

    return Response(_serialize_plan_request(plan_request))
