"""
Owner-controlled per-member feature access, bounded by the company's
Plan (RAG.models.FEATURE_CODES). Composes with, never replaces,
org_permission_service.py (org rank checks) and billing_service.py
(Plan-level feature codes) - has_feature_access() is the ONE function
every call site (DRF permission class, member serializer, sidebar
visibility) should go through, so the composition rule (platform
bypass -> Plan gate -> member override) is expressed exactly once.
"""

from ..models import FEATURE_CODES, OrganizationMembership
from . import billing_service
from .org_audit_log_service import log_org_activity
from .org_permission_service import get_org_membership


class FeatureAccessError(Exception):
    pass


def has_feature_access(user, organization, feature_code):
    """
    organization=None means no organization - unreachable in practice
    since Personal Workspace was removed as an account type (every
    account now belongs to a Company), kept as a defensive True
    (unrestricted) rather than a crash if ever hit.

    Fail-closed exception (Company branch only): a user with no active
    membership in this organization (and no platform bypass) always
    gets False, even if the org's Plan includes the feature - matching
    get_org_membership()'s existing "no active membership = no access"
    contract.
    """
    from .permission_service import user_has_permission
    if user_has_permission(user, "organizations.manage"):
        return True

    if organization is None:
        return True

    if not billing_service.org_has_plan_feature(organization, feature_code):
        return False

    membership = get_org_membership(user, organization)
    if membership is None:
        return False

    return feature_code not in (membership.disabled_features or [])


def get_member_feature_access(organization, membership):
    """{feature_code: bool} for every FEATURE_CODES entry - composes the org's Plan with this
    member's own override, exactly what the toggle matrix UI renders (checked/disabled state
    per checkbox). An Owner is never restricted, matching set_member_disabled_features()'s guard."""
    plan_codes = billing_service.org_plan_feature_codes(organization)
    is_owner = membership.role == OrganizationMembership.Role.OWNER
    disabled = set(membership.disabled_features or [])
    return {
        code: (plan_codes is None or code in plan_codes) and (is_owner or code not in disabled)
        for code in FEATURE_CODES
    }


def set_member_disabled_features(organization, actor_membership, target_membership, disabled_codes, request=None):
    """
    Owner-only - the caller enforces the coarse gate via
    HasOrgPermission("members.update_role") the same way
    org_membership_service.py's other member-mutating actions do.
    Refuses to target an Owner - an Owner's own feature access is never
    restrictable by this system (no legitimate reason to restrict full
    org-management rank, and it risks an unrecoverable self-lockout).
    `disabled_codes` replaces the full set (same "send the whole list"
    shape role assignment uses), and can only ever narrow access - it
    is validated against FEATURE_CODES but never checked against the
    Plan here, since org_has_plan_feature()/has_feature_access() apply
    the Plan ceiling at read time regardless of what this list contains.
    """
    if target_membership.organization_id != organization.id:
        raise FeatureAccessError("That member does not belong to this organization.")
    if target_membership.role == OrganizationMembership.Role.OWNER:
        raise FeatureAccessError("An Owner's feature access can't be restricted.")

    unknown = set(disabled_codes or []) - set(FEATURE_CODES)
    if unknown:
        raise FeatureAccessError(f"Unknown feature code(s): {', '.join(sorted(unknown))}")

    target_membership.disabled_features = sorted(set(disabled_codes or []))
    target_membership.save(update_fields=["disabled_features"])

    log_org_activity(
        organization=organization,
        actor=actor_membership.user,
        action="member.feature_access_updated",
        description=f"{actor_membership.user.username} updated {target_membership.user.username}'s feature access",
        metadata={"target_user_id": target_membership.user_id, "disabled_features": target_membership.disabled_features},
        request=request,
    )
    return target_membership
