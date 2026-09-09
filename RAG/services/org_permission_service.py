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
    of every organization *for rank-comparison purposes only* - e.g.
    dashboard_views.py's scope_to_own, which decides whether an
    oversight viewer sees org-wide vs. own-only aggregate numbers. It
    does NOT by itself grant the org-management codename surface
    (members/settings/billing/audit-logs) - user_has_org_permission()/
    get_org_permission_codenames() deliberately do NOT derive their
    answer from this function's OWNER fiction; they re-check real
    membership themselves and, absent one, grant only
    BYPASS_ONLY_CODENAMES (privacy: platform oversight can see a
    company's stats and suspend/delete it, but should not be able to
    read its members/settings/billing/audit log just by clicking in
    from Admin > Companies). Never a real OrganizationMembership row -
    purely a computed answer, so it can't be mistaken for one anywhere
    that queries the table directly.
    """

    membership = get_org_membership(user, organization)
    if membership:
        return membership.role

    from .permission_service import user_has_permission
    if user_has_permission(user, "organizations.manage"):
        return OWNER

    return None


# Codenames a platform-wide oversight bypass (Admin/Super Admin via the
# "organizations.manage" platform permission, never a real
# OrganizationMembership row - see get_user_org_role()'s docstring) may
# exercise on a company it doesn't belong to. Deliberately just the
# read-only stats snapshot ("Admin > Companies" click-through) - NOT
# the member-management/settings/billing/audit-log surface a real
# Owner gets, even though get_user_org_role() still answers OWNER for
# rank-comparison purposes elsewhere (e.g. dashboard_views.py's
# scope_to_own, which decides whether an oversight viewer sees
# org-wide vs. own-only numbers - that's a read-only aggregate too, so
# it's fine for it to keep trusting the OWNER-shaped rank). Suspending/
# deleting an organization is a SEPARATE platform-level action
# (platform_organization_action_view, gated directly on
# HasPagePermission("organizations.manage")) that never goes through
# this org-scoped codename table at all, so it's unaffected by this
# restriction - platform oversight can still view a company's stats
# and suspend/delete it, just never reach into its members, settings,
# billing, or audit log the way that company's own Owner can.
BYPASS_ONLY_CODENAMES = frozenset({"organization.view"})


def user_has_org_permission(user, organization, codename):
    min_role = ORG_PERMISSION_MIN_ROLE.get(codename)
    if min_role is None:
        logger.warning("Unknown organization permission codename requested: %s", codename)
        return False

    membership = get_org_membership(user, organization)
    if membership:
        return ROLE_RANK[membership.role] >= ROLE_RANK[min_role]

    from .permission_service import user_has_permission
    return codename in BYPASS_ONLY_CODENAMES and user_has_permission(user, "organizations.manage")


def get_org_permission_codenames(user, organization):
    """Every org-permission codename `user` holds in `organization`, sorted - the org-scoped counterpart to permission_service.get_user_permission_codenames(), same JSON-serializable-list shape for frontend/template consumption. A platform oversight bypass (no real membership) gets only BYPASS_ONLY_CODENAMES, never the full role-derived set a real Owner/Member gets."""

    membership = get_org_membership(user, organization)
    if membership:
        return ORG_ROLE_PERMISSIONS.get(membership.role, [])

    from .permission_service import user_has_permission
    if user_has_permission(user, "organizations.manage"):
        return sorted(BYPASS_ONLY_CODENAMES)
    return []


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


def user_can_view_org_analytics(user, organization, feature_code=None):
    """
    Shared by Analytics and Reports: True for Personal Workspace
    (organization=None - there it's just this user's own usage, same
    as every other page), or whenever get_user_org_role() answers
    OWNER for `organization` - which already covers both a real Owner
    (the same org-management rank that grants members/billing/settings
    everywhere else now extends to Analytics/Reports too - there was
    no real reason for those two pages to be the one place an Owner
    doesn't hold their own company's complete surface) AND a platform
    Admin's oversight bypass (get_user_org_role() treats
    "organizations.manage" as OWNER-equivalent for exactly this kind of
    rank check - see its own docstring).

    A plain Member holds none of the org-management surface, so they
    never get the Owner's whole-team view here - but the org's own
    Owner can still individually grant a Member Analytics/Reports via
    the ordinary per-member feature toggle
    (org_member_feature_service.has_feature_access(), itself bounded by
    the company's Plan), same as every other FEATURE_CODES entry. Pass
    `feature_code` ("analytics" or "reports") to honor that grant; the
    caller (analytics_views.py/reports_views.py) is then responsible
    for scoping what a Member let in this way actually sees down to
    their own activity only (scope_to_own) - this function only
    answers "may they view the page at all," never "how much of the
    company's data should they see." Omitting `feature_code` keeps the
    strict Owner-only answer, for any future caller that shouldn't
    honor the per-member grant.
    """

    if organization is None:
        return True
    if get_user_org_role(user, organization) == OWNER:
        return True
    if feature_code is None:
        return False

    from .org_member_feature_service import has_feature_access
    return has_feature_access(user, organization, feature_code)


def resolve_admin_scoped_organization(request):
    """
    Analytics/Reports-style organization resolution: a platform Admin
    may override the usual X-Organization-Slug header via an explicit
    ?organization=<slug> (or "personal") query param, to view/audit
    ANY company's Analytics/Reports - "overall" (Personal/platform)
    numbers via ?organization=personal, or one specific company's own
    numbers via its slug - without first switching their active
    workspace to it, which they usually can't do at all (an Admin is
    rarely a real member of the company they need to look at). Every
    other viewer (that company's own Owner, browsing their own active
    workspace) is completely unaffected by this - the override is only
    ever honored for permission_service.is_admin(), and falls straight
    through to the ordinary header-based resolve_request_organization()
    otherwise. Returns (organization, is_admin_override) - callers use
    the second value to tell "an Admin is deliberately auditing
    someone else's company" apart from the ordinary case, both of
    which can resolve organization=None.

    request.GET (not request.query_params) so this works identically
    from both a DRF view and a plain-Django one (@api_view vs
    @permission_required, e.g. reports_views.py's CSV export views) -
    DRF's Request forwards unknown attributes to the wrapped
    HttpRequest, so .GET exists either way, unlike .query_params.
    """

    from .permission_service import is_admin

    org_param = request.GET.get("organization")
    if org_param and is_admin(request.user):
        if org_param == "personal":
            return None, True
        organization = Organization.objects.filter(slug=org_param, status=Organization.Status.ACTIVE).first()
        if organization is None:
            raise Http404("Organization not found.")
        return organization, True

    organization, _ = resolve_request_organization(request)
    return organization, False
