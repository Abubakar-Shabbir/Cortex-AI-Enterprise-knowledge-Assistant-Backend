"""
Owner-controlled per-member usage sub-quotas - a hard, count-based
ceiling on how many questions/AI Task runs ONE member may use in the
current billing period, kept deliberately separate from the
organization's own AI credit balance (billing_service.py's SOLE
org-wide spend-metering currency - see that module's docstring).
Structurally parallel to org_member_feature_service.py
(set_member_disabled_features()): the read side
(billing_service.check_member_query_limit()/
check_member_ai_task_limit()/get_member_usage()) lives in
billing_service.py since it shares that module's period/window math;
set_member_usage_limits() below is the one write path.
"""

from ..models import OrganizationMembership
from .billing_service import get_active_subscription
from .org_audit_log_service import log_org_activity


class MemberLimitError(Exception):
    pass


def _parse_limit(value, plan_ceiling, label):
    """
    None/empty means "no member-specific cap" (the field's own
    default) - anything else must be a non-negative whole number, and
    when the organization's active Plan defines its own quota for this
    dimension, a member's cap can only narrow it, never exceed it: a
    per-member limit is a SUB-quota of the Plan's own total, not a
    second, potentially-larger axis (see OrganizationMembership.
    max_queries_per_month's help_text).
    """
    if value in (None, "", "null"):
        return None
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise MemberLimitError(f"{label} must be a whole number.")
    if value < 0:
        raise MemberLimitError(f"{label} must be a non-negative whole number.")
    if plan_ceiling is not None and value > plan_ceiling:
        raise MemberLimitError(f"{label} can't exceed this organization's own plan limit ({plan_ceiling}).")
    return value


def set_member_usage_limits(organization, actor_membership, target_membership, max_queries_per_month, max_ai_task_runs_per_month, request=None):
    """
    Owner-only - the caller enforces the coarse gate via
    HasOrgPermission("members.update_role"), the same convention
    set_member_disabled_features() uses. Refuses to target an Owner -
    an Owner's own usage is never restrictable by this system, same
    rationale as feature access (no legitimate reason to restrict full
    org-management rank, and it risks an unrecoverable self-lockout if
    an org's only Owner accidentally capped themselves to 0).
    """
    if target_membership.organization_id != organization.id:
        raise MemberLimitError("That member does not belong to this organization.")
    if target_membership.role == OrganizationMembership.Role.OWNER:
        raise MemberLimitError("An Owner's usage can't be restricted.")

    subscription = get_active_subscription(organization)
    plan = subscription.plan if subscription else None

    max_queries_per_month = _parse_limit(
        max_queries_per_month, plan.max_queries_per_month if plan else None, "Query limit",
    )
    max_ai_task_runs_per_month = _parse_limit(
        max_ai_task_runs_per_month, plan.max_ai_task_runs_per_month if plan else None, "AI Task run limit",
    )

    target_membership.max_queries_per_month = max_queries_per_month
    target_membership.max_ai_task_runs_per_month = max_ai_task_runs_per_month
    target_membership.save(update_fields=["max_queries_per_month", "max_ai_task_runs_per_month"])

    log_org_activity(
        organization=organization,
        actor=actor_membership.user,
        action="member.usage_limits_updated",
        description=f"{actor_membership.user.username} updated {target_membership.user.username}'s usage limits",
        metadata={
            "target_user_id": target_membership.user_id,
            "max_queries_per_month": max_queries_per_month,
            "max_ai_task_runs_per_month": max_ai_task_runs_per_month,
        },
        request=request,
    )
    return target_membership
