"""
Personal-Workspace counterpart of billing_views.py - a Personal user's
own view into (and self-service request/top-up for) their Personal
Plan, completely separate from any Company org's billing (see
billing_service.py's module docstring). There's no HasOrgPermission-
equivalent needed here since there's no membership/org concept for a
Personal account - every view below is IsAuthenticated plus an inline
account_type check, mirroring the same "PERSONAL vs COMPANY is decided
once, never re-decided by the client" rule
org_permission_service.resolve_request_organization() enforces for
company-scoped endpoints.
"""

from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from ..models import Plan, PlanChangeRequest, UserProfile
from ..services import billing_service
from ..services.billing_service import BillingServiceError
from .billing_views import _serialize_plan, _serialize_plan_request


def _require_personal_account(request):
    """Returns an error Response if this account isn't Personal, else None."""
    profile = getattr(request.user, "profile", None)
    if profile is None or profile.account_type != UserProfile.AccountType.PERSONAL:
        return Response({"error": "This endpoint is only available to Personal Workspace accounts."}, status=403)
    return None


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def personal_billing_view(request):
    denied = _require_personal_account(request)
    if denied:
        return denied

    usage = billing_service.get_personal_usage(request.user)
    available_plans = Plan.objects.filter(plan_type=Plan.PlanType.PERSONAL, is_active=True)
    pending_request = PlanChangeRequest.objects.filter(
        user=request.user, status=PlanChangeRequest.Status.PENDING,
    ).select_related("requested_plan").first()

    return Response({
        **usage,
        "available_plans": [_serialize_plan(p) for p in available_plans],
        "pending_request": _serialize_plan_request(pending_request) if pending_request else None,
    })


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def personal_plan_request_view(request):
    denied = _require_personal_account(request)
    if denied:
        return denied

    plan = get_object_or_404(Plan, id=request.data.get("plan_id"), plan_type=Plan.PlanType.PERSONAL)
    try:
        plan_request = billing_service.request_plan_change(plan, request.user, user=request.user, request=request)
    except BillingServiceError as e:
        return Response({"error": str(e)}, status=400)
    return Response(_serialize_plan_request(plan_request), status=201)


@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def personal_ai_credits_view(request):
    denied = _require_personal_account(request)
    if denied:
        return denied

    profile = request.user.profile

    if request.method == "GET":
        transactions = request.user.personal_ai_credit_transactions.select_related("actor")[:50]
        return Response({
            "balance": profile.ai_credits_balance,
            "transactions": [
                {
                    "id": t.id, "amount": t.amount, "balance_after": t.balance_after,
                    "reason": t.reason, "actor": t.actor.username if t.actor_id else None,
                    "created_at": t.created_at.isoformat(),
                }
                for t in transactions
            ],
        })

    subscription = billing_service.get_active_personal_subscription(request.user)
    if subscription is None or not subscription.plan.allow_credit_purchase:
        return Response({"error": "Your current plan doesn't allow purchasing extra credits."}, status=400)

    try:
        amount = int(request.data.get("amount"))
    except (TypeError, ValueError):
        return Response({"error": "Enter a positive whole number."}, status=400)

    try:
        balance = billing_service.add_personal_ai_credits(request.user, amount, request.user)
    except BillingServiceError as e:
        return Response({"error": str(e)}, status=400)

    transactions = request.user.personal_ai_credit_transactions.select_related("actor")[:50]
    return Response({
        "balance": balance,
        "transactions": [
            {
                "id": t.id, "amount": t.amount, "balance_after": t.balance_after,
                "reason": t.reason, "actor": t.actor.username if t.actor_id else None,
                "created_at": t.created_at.isoformat(),
            }
            for t in transactions
        ],
    })
