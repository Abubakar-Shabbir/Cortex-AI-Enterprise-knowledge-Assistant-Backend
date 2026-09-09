"""
Organization Member Management - remove/reassign-role/suspend a member
within one organization. Every mutation here routes through
org_permission_service's privilege-escalation guards before touching
the database - this module is the one place those guards actually get
enforced against a real membership row, the same relationship
RAG/views.py's admin_users_view has with permission_service's
platform-level guards.

None of these functions re-check "does the actor have members.*
permission at all" - that coarse gate is the calling view's job (via
org_permission_required), same "coarse gate + fine-grained internal
scoping" split the rest of this codebase already uses (see
permission_service.has_any_settings_permission's docstring for the
canonical statement of that pattern). What this module owns is the
guard a bare permission check can't express: not just "can Admins
change roles" but "can THIS Admin change THIS particular membership".
"""

from ..models import OrganizationMembership
from .org_audit_log_service import log_org_activity
from .org_permission_service import (
    can_actor_assign_org_role,
    can_actor_manage_org_member,
    is_last_org_owner,
)


class MembershipError(Exception):
    """Raised for every member-management failure a caller should show back to the user (privilege escalation, last-Owner lockout, ...) - same one-exception-type-per-module shape as InvitationError/OrganizationServiceError."""


def list_members(organization, status=None):
    qs = OrganizationMembership.objects.filter(organization=organization).select_related("user", "user__profile")
    if status:
        qs = qs.filter(status=status)
    return qs.order_by("role", "user__username")


def update_member_role(organization, actor_membership, target_membership, new_role, request=None):
    """
    Reassign `target_membership.role` to `new_role`. Guarded twice,
    for two different escalation shapes: can_actor_manage_org_member
    (may this actor touch this target at all - never a peer/superior)
    AND can_actor_assign_org_role (may this actor hand out this
    SPECIFIC new role - never Owner unless the actor is already an
    Owner, never a role at or above the actor's own rank). A demotion
    away from the org's last Owner is blocked separately, by
    is_last_org_owner, so the organization can never end up with zero
    Owners.
    """

    if target_membership.organization_id != organization.id:
        raise MembershipError("That member does not belong to this organization.")

    if not can_actor_manage_org_member(actor_membership.role, target_membership.role):
        raise MembershipError("You don't have permission to change this member's role.")

    if not can_actor_assign_org_role(actor_membership.role, new_role):
        raise MembershipError("You don't have permission to assign that role.")

    if target_membership.role == OrganizationMembership.Role.OWNER and new_role != OrganizationMembership.Role.OWNER:
        if is_last_org_owner(target_membership.user, organization):
            raise MembershipError("This is the last Owner - promote another member to Owner first.")

    old_role = target_membership.role
    target_membership.role = new_role
    update_fields = ["role"]
    if new_role == OrganizationMembership.Role.OWNER and target_membership.disabled_features:
        # An Owner is never restricted (has_feature_access()/get_member_feature_access()'s
        # contract) - clear now rather than leave a stale list that would silently
        # reactivate if this Owner is ever demoted back to Member later.
        target_membership.disabled_features = []
        update_fields.append("disabled_features")
    target_membership.save(update_fields=update_fields)

    log_org_activity(
        organization=organization,
        actor=actor_membership.user,
        action="member.role_changed",
        description=f"{actor_membership.user.username} changed {target_membership.user.username}'s role from {old_role} to {new_role}",
        metadata={"target_user_id": target_membership.user_id, "old_role": old_role, "new_role": new_role},
        request=request,
    )

    return target_membership


def remove_member(organization, actor_membership, target_membership, request=None):
    """
    Remove `target_membership` from `organization` entirely (hard
    delete of the membership row, not a status flip - re-joining later
    requires a fresh registration/invitation). Blocked for the
    organization's last Owner, same as a role change away from Owner.

    2026-09-06: also deactivates the removed user's ACCOUNT
    (`User.is_active = False`), not just their membership - a company-
    registered member (see org_member_registration_service.py) has no
    Personal Workspace or independent existence outside the company
    that registered them, so being removed blocks them from logging in
    anywhere at all, immediately (Django's own auth backend already
    refuses `is_active=False` at authenticate() time - no separate
    session-invalidation step is needed). Deliberately reversible, not
    permanent: org_member_registration_service.register_company_member()
    reactivates this same account (rather than creating a duplicate)
    if this same email is registered into a company again later.
    """

    if target_membership.organization_id != organization.id:
        raise MembershipError("That member does not belong to this organization.")

    if not can_actor_manage_org_member(actor_membership.role, target_membership.role):
        raise MembershipError("You don't have permission to remove this member.")

    if target_membership.role == OrganizationMembership.Role.OWNER and is_last_org_owner(target_membership.user, organization):
        raise MembershipError("This is the last Owner - promote another member to Owner first, or delete the organization instead.")

    removed_user = target_membership.user
    removed_username = removed_user.username
    removed_user_id = target_membership.user_id
    target_membership.delete()

    removed_user.is_active = False
    removed_user.save(update_fields=["is_active"])

    log_org_activity(
        organization=organization,
        actor=actor_membership.user,
        action="member.removed",
        description=f"{actor_membership.user.username} removed {removed_username} from the organization",
        metadata={"target_user_id": removed_user_id},
        request=request,
    )


def set_member_status(organization, actor_membership, target_membership, status, request=None):
    """Suspend/reactivate a member without removing them - a suspended membership is treated identically to no membership at all by org_permission_service.get_org_membership(), so this immediately revokes (or restores) access. Same last-Owner protection as remove_member() for a suspension."""

    if target_membership.organization_id != organization.id:
        raise MembershipError("That member does not belong to this organization.")

    if not can_actor_manage_org_member(actor_membership.role, target_membership.role):
        raise MembershipError("You don't have permission to change this member's status.")

    if (
        status == OrganizationMembership.Status.SUSPENDED
        and target_membership.role == OrganizationMembership.Role.OWNER
        and is_last_org_owner(target_membership.user, organization)
    ):
        raise MembershipError("This is the last Owner - promote another member to Owner first.")

    target_membership.status = status
    target_membership.save(update_fields=["status"])

    log_org_activity(
        organization=organization,
        actor=actor_membership.user,
        action="member.suspended" if status == OrganizationMembership.Status.SUSPENDED else "member.reactivated",
        description=f"{actor_membership.user.username} set {target_membership.user.username}'s status to {status}",
        metadata={"target_user_id": target_membership.user_id},
        request=request,
    )

    return target_membership
