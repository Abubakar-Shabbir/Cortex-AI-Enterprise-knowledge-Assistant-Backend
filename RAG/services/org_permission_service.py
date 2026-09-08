"""
Organization Permission Service

Single source of truth for organization-scoped role/permission checks -
the tenant-level counterpart to RAG.services.permission_service, which
remains the platform-level source of truth and is untouched by this
module. The two are deliberately kept separate rather than merged into
one generalized "role with a scope" system: a platform Admin and an
organization's Owner/Admin are different concepts (different tables,
different permission namespaces, different decorators) that happen to
look structurally similar. Merging them would tempt every future check
into ambiguous "which scope is this?" bugs - exactly the class of bug
this module's isolation guarantees exist to prevent.

There is no "Super Admin" role for organization oversight. The
platform Admin role already has every permission by design
(Role.has_permission's bypass in RAG/models.py), so platform-wide
organization management (view/create/suspend/delete any organization)
is expressed as new "organizations.*" codenames in the EXISTING
platform Permission/Role system (see seed_rbac.py), which Admin's
"__all__" grant already covers - not a second top-tier platform role.

Unlike platform RBAC's fully dynamic Role model, an organization's
`role` (OrganizationMembership.role) is a fixed two-tier CharField
(Owner / Member, collapsed 2026-09-06 from an earlier four-tier
Owner/Admin/Manager/Member ladder - a Member holds zero org-management
permissions now, full stop) - ORG_ROLE_PERMISSIONS below is the one
place that maps each tier to
what it grants. If per-org custom roles are ever needed, the migration
path is to promote OrganizationMembership.role into an OrgRole FK with
its own M2M to a parallel OrgPermission model, mirroring platform
RBAC's shape exactly - nothing in this module's public API
(get_org_membership, user_has_org_permission, the privilege-escalation
guards) would need to change shape for callers, only what backs it
internally.

SECURITY: every function here that takes an `organization` must have
already been resolved from a slug/id via a lookup a caller trusts
(typically resolve_organization_context() below) - never trust a
client-supplied numeric id/slug as proof of anything beyond "which
organization is being asked about." Membership, role, and permissions
are always re-derived server-side from OrganizationMembership.
"""

import logging

from django.http import Http404

from ..models import (
    ORG_ROLE_MEMBER,
    ORG_ROLE_OWNER,
    Organization,
    OrganizationMembership,
    UserProfile,
)

logger = logging.getLogger(__name__)

OWNER = ORG_ROLE_OWNER
MEMBER = ORG_ROLE_MEMBER

# Two tiers only (2026-09-06 product decision - see OrganizationMembership's
# docstring): Owner holds full organization control, Member holds none
# of the org-management surface at all. The privilege-escalation guards
# below still reduce to "is my rank strictly higher than the rank I'm
# trying to act on/assign", same shape as when this was a four-tier
# ladder - collapsing the ladder to two rungs didn't need to change
# that logic, only the table it ranks.
ROLE_RANK = {
    MEMBER: 0,
    OWNER: 1,
}

# codename -> minimum role tier required. Every codename here is
# OWNER-only by design: a plain Member gets zero access to the
# organization-management surface (no Members page, no Settings, no
# Billing, no Audit Log, not even read-only Overview) - they operate
# entirely within the ordinary app portal (Documents, Ask AI, ...)
# scoped to their company via X-Organization-Slug, never through this
# permission table at all. ORG_ROLE_PERMISSIONS[MEMBER] is therefore
# always the empty list - that emptiness is the point, not a gap.
ORG_PERMISSION_MIN_ROLE = {
    "organization.view": OWNER,
    "organization.update": OWNER,
    "organization.delete": OWNER,
    "members.view": OWNER,
    "members.invite": OWNER,
    "members.remove": OWNER,
    "members.update_role": OWNER,
    "settings.view": OWNER,
    "settings.update": OWNER,
    "audit_logs.view": OWNER,
    "billing.view": OWNER,
    "ai_credits.view": OWNER,
    "ai_credits.manage": OWNER,
    # Metadata only (who/when/confidence/method) - never the actual
    # question/answer text, which this codename does NOT grant. See
    # organizations_views.organization_queries_view's docstring for
    # the "logs visible, content protected" split this backs.
    "queries.view": OWNER,
}

# role -> sorted list of codenames it grants, derived from
# ORG_PERMISSION_MIN_ROLE rather than duplicated - the frontend-facing
# form (see get_org_permission_codenames), same "compute the
# permission set exactly one way" principle permission_service.py's
# get_user_permission_set follows for platform RBAC.
ORG_ROLE_PERMISSIONS = {
    role: sorted(
        codename
        for codename, min_role in ORG_PERMISSION_MIN_ROLE.items()
        if ROLE_RANK[role] >= ROLE_RANK[min_role]
    )
    for role in ROLE_RANK
}


def get_org_membership(user, organization):
    """
    The ACTIVE OrganizationMembership row for `user` in `organization`,
    or None. This is the one function every other check in this module
    (and every org-scoped view) ultimately goes through - a suspended
    membership is treated identically to no membership at all, so
    suspending a member immediately revokes access without a separate
    "is this membership active" check scattered at every call site.
    """

    if not user or not getattr(user, "is_authenticated", False) or not organization:
        return None

    return OrganizationMembership.objects.select_related("organization").filter(
        user=user, organization=organization, status=OrganizationMembership.Status.ACTIVE
    ).first()


def get_user_org_role(user, organization):
    """
    Real membership role if `user` actually belongs to `organization`.
    Otherwise, a platform-wide "organizations.manage" holder (Admin via
    its bypass, or the dynamic Super Admin role) is treated as an OWNER
    of every organization - the same trust `platform_organization_action_view`
    already extends for suspend/delete, now applied uniformly to every
    org-scoped view/permission check that goes through this function
    (organization.view, members.view, billing.view,
    audit_logs.view, ...). This is what lets clicking into a company
    from Admin > Companies actually open its pages instead of 403ing -
    Admin/Super Admin were never members of any company they didn't
    personally create. Never a real OrganizationMembership row - purely
    a computed answer, so it can't be mistaken for one anywhere that
    queries the table directly.
    """

    membership = get_org_membership(user, organization)
    if membership:
        return membership.role

    from .permission_service import user_has_permission
    if user_has_permission(user, "organizations.manage"):
        return OWNER

    return None


def user_has_org_permission(user, organization, codename):
    role = get_user_org_role(user, organization)
    if not role:
        return False
    min_role = ORG_PERMISSION_MIN_ROLE.get(codename)
    if min_role is None:
        logger.warning("Unknown organization permission codename requested: %s", codename)
        return False
    return ROLE_RANK[role] >= ROLE_RANK[min_role]


def get_org_permission_codenames(user, organization):
    """Every org-permission codename `user` holds in `organization`, sorted - the org-scoped counterpart to permission_service.get_user_permission_codenames(), same JSON-serializable-list shape for frontend/template consumption."""

    role = get_user_org_role(user, organization)
    return ORG_ROLE_PERMISSIONS.get(role, [])


def get_user_organizations(user):
    """Every ACTIVE organization `user` belongs to, with their membership - powers the workspace switcher. Never includes a suspended membership or a non-active organization."""

    if not user or not getattr(user, "is_authenticated", False):
        return OrganizationMembership.objects.none()

    return (
        OrganizationMembership.objects.select_related("organization", "organization__org_type")
        .filter(user=user, status=OrganizationMembership.Status.ACTIVE, organization__status=Organization.Status.ACTIVE)
        .order_by("organization__name")
    )


def resolve_organization_context(user, org_slug):
    """
    The one function org-scoped views/decorators should call to turn a
    URL's org_slug into a trusted (organization, membership) pair.
    Raises Http404 for a slug that doesn't resolve to an active
    organization (never leaks "this org exists but you can't see it"
    vs. "this org doesn't exist" - both look like a 404), and returns
    (organization, None) for an authenticated user with no/suspended
    membership so the caller can choose the right response (403 vs. an
    "invite only" message) rather than this function assuming one.
    """

    organization = Organization.objects.filter(
        slug=org_slug, status=Organization.Status.ACTIVE
    ).first()

    if not organization:
        raise Http404("Organization not found.")

    membership = get_org_membership(user, organization)

    return organization, membership


# Header a request carries to say "I'm currently working inside this
# organization" - read by resolve_request_organization() below for
# every EXISTING, shared endpoint (Documents, and eventually Ask AI/
# AI Tasks/Knowledge Graph) that isn't itself a dedicated org-scoped
# URL (those use resolve_organization_context() via org_slug directly
# instead - see organizations_views.py). A header rather than a query
# param/URL segment here specifically because these are pre-existing
# routes this multi-tenancy build must not fork into "/documents/" and
# "/organizations/<slug>/documents/" duplicate URL trees just to add
# workspace-switching - the frontend's OrganizationContext sets this
# header on every request once, rather than every one of dozens of
# existing call sites growing an org_slug URL/query parameter.
ORGANIZATION_HEADER = "X-Organization-Slug"


def resolve_request_organization(request):
    """
    The header-based counterpart to resolve_organization_context() for
    a pre-existing, shared endpoint rather than a dedicated org-scoped
    URL.

    Real OrganizationMembership is always the actual authorization
    source - identical to every dedicated org-scoped URL's own
    resolve_organization_context() call, which has never looked at
    account_type at all. This function's only extra job is choosing
    the right DEFAULT when the request carries no header, since
    "Personal vs Company is decided once at signup, never by a
    workspace switcher" (UserProfile.account_type's help_text) means
    that default must never silently offer a scope the account isn't
    supposed to work in day-to-day:

    - Header PRESENT -> resolve and re-verify membership exactly like
      resolve_organization_context() does, regardless of account_type
      (never trust the header beyond "which organization is being
      asked about"). A present-but-invalid header (unknown/suspended
      org, or a real org this user isn't an active member of) raises
      rather than silently falling back to Personal Workspace - a
      client that explicitly asked for Org A must never be silently
      shown something else just because that request was
      misauthorized; that would look like data loss, not an access
      error.
    - Header ABSENT, account_type=PERSONAL -> (None, None), Personal
      Workspace - a Personal account's frontend never even shows a
      workspace switcher (see Sidebar.jsx's WorkspaceSwitcher), so it
      never sends this header in normal use; this is just the default
      for a bare request.
    - Header ABSENT, account_type=COMPANY -> defaults to one of the
      account's own organizations (deterministic, alphabetical - same
      ordering the workspace-switcher list itself uses) rather than
      Personal Workspace, since a Company account has no Personal
      Workspace to show; a Company account genuinely holding no
      organization at all (e.g. removed from its last one) degrades to
      (None, None) rather than erroring.
    """

    from django.core.exceptions import PermissionDenied

    org_slug = request.headers.get(ORGANIZATION_HEADER)

    if org_slug:
        organization, membership = resolve_organization_context(request.user, org_slug)
        if not membership:
            raise PermissionDenied("You don't have access to this organization.")
        return organization, membership

    profile = getattr(request.user, "profile", None)
    account_type = getattr(profile, "account_type", UserProfile.AccountType.PERSONAL)

    if account_type != UserProfile.AccountType.COMPANY:
        return None, None

    membership = get_user_organizations(request.user).first()
    return (membership.organization, membership) if membership else (None, None)


# ============================================================
# Privilege-escalation guards
# ============================================================
#
# ORG_PERMISSION_MIN_ROLE above answers "can this role do X at all
# within its own organization". These answer the question a per-
# permission check alone can't: "does this specific action, against
# this specific target membership, hand out or exercise more authority
# than the actor legitimately holds in THIS organization". Mirrors
# permission_service.py's own privilege-escalation guard section -
# same shape, organization-scoped instead of platform-scoped.
#
# The invariants these enforce, matching the platform-level ones:
#   1. An actor can only assign a role strictly below their own rank -
#      never a peer or superior role, never themselves upward.
#   2. An actor can only act on (remove/suspend/reassign) a member
#      strictly below their own rank - never a peer or superior member.
#   3. No action may leave an organization with zero Owners.


def can_actor_assign_org_role(actor_role, target_role):
    """
    True if a member holding `actor_role` may assign `target_role` to
    someone in the same organization. An Owner may assign any role,
    including Owner itself - that's the only way organization
    ownership is ever transferred or shared. A Member can never assign
    anything (there is no rank below Member to promote someone from,
    and the rank comparison below already falls out to False for it
    without needing to special-case Member by name).
    """

    if actor_role == OWNER:
        return True

    if target_role not in ROLE_RANK or actor_role not in ROLE_RANK:
        return False

    return ROLE_RANK[actor_role] > ROLE_RANK[target_role]


def get_assignable_org_roles(actor_role):
    """The role values `actor_role` is allowed to assign to someone else, in ascending rank order - the frontend-facing counterpart to can_actor_assign_org_role(), so a role picker never even lists a role the actor isn't allowed to grant."""

    return [role for role in (MEMBER, OWNER) if can_actor_assign_org_role(actor_role, role)]


def can_actor_manage_org_member(actor_role, target_role):
    """
    True if a member holding `actor_role` may remove/suspend/reassign
    a member holding `target_role`. Strictly rank-based: an actor may
    only manage a member ranked below them - never a peer, never
    someone above them. An Owner may manage anyone, including another
    Owner (subject to the last-Owner check below, enforced separately
    by the caller before this ever executes a removal).
    """

    if actor_role == OWNER:
        return True

    if target_role not in ROLE_RANK or actor_role not in ROLE_RANK:
        return False

    return ROLE_RANK[actor_role] > ROLE_RANK[target_role]


def is_last_org_owner(user, organization):
    """
    True if `user` holds the Owner role in `organization` and is the
    only ACTIVE Owner there. The trigger for blocking a removal,
    suspension, or role-reassignment away from Owner - without this,
    an organization could be left with zero Owners, an unrecoverable
    lockout for that tenant short of direct database access. Mirrors
    permission_service.is_last_admin exactly, scoped to one
    organization instead of the whole platform.
    """

    membership = get_org_membership(user, organization)

    if not membership or membership.role != OWNER:
        return False

    return OrganizationMembership.objects.filter(
        organization=organization, role=OWNER, status=OrganizationMembership.Status.ACTIVE
    ).count() <= 1
