"""
DRF permission classes wrapping RAG.services.permission_service - the
same RBAC checks RAG/decorators.py enforces on the classic Django
views, so an API endpoint backing a React page is gated exactly the
same way its Django-template predecessor was. No new authorization
logic is introduced here, only a DRF-shaped adapter over the existing
one.
"""

from rest_framework.permissions import BasePermission

from ..models import ORG_ROLE_OWNER, OrganizationMembership
from ..services.org_member_feature_service import has_feature_access
from ..services.permission_service import has_admin_area_access, user_has_permission
from ..services.org_permission_service import resolve_organization_context, resolve_request_organization, user_has_org_permission


class HasAdminAreaAccess(BasePermission):
    """
    DRF mirror of RAG.decorators.admin_area_required - coarse,
    role-based "can this account reach the admin area at all" gate
    (permission_service.has_admin_area_access), not a specific
    permission codename. Used by the Admin Overview API endpoint, the
    same population admin_dashboard_view's Django-template predecessor
    is restricted to.
    """

    message = "You don't have access to this page."

    def has_permission(self, request, view):
        return bool(
            request.user
            and request.user.is_authenticated
            and has_admin_area_access(request.user)
        )


def HasPagePermission(*codenames):
    """
    Factory mirroring RAG.decorators.permission_required(*codenames) -
    every codename must be granted by the requester's role. Usage:
    permission_classes = [HasPagePermission("pages.documents")]
    """

    class _HasPagePermission(BasePermission):
        message = "You don't have access to this resource."

        def has_permission(self, request, view):
            return bool(
                request.user
                and request.user.is_authenticated
                and all(user_has_permission(request.user, code) for code in codenames)
            )

    return _HasPagePermission


def HasOrgPermission(*codenames):
    """
    Factory mirroring RAG.decorators.org_permission_required(*codenames) -
    the DRF-side counterpart for an organization-scoped API view whose
    URL includes `org_slug`. Resolves membership fresh from
    OrganizationMembership via org_permission_service - never trusts
    org_slug beyond "which organization is being asked about" - and
    attaches request.organization / request.org_membership on success,
    exactly like the decorator does for classic Django views, so both
    surfaces expose the same trusted attributes to the view body.

    Usage: permission_classes = [HasOrgPermission("members.invite")]
    (the view's URL must supply `org_slug` as a kwarg, same convention
    DRF already uses for every other path-parameterized viewset here).
    """

    class _HasOrgPermission(BasePermission):
        message = "You don't have access to this organization."

        def has_permission(self, request, view):
            if not (request.user and request.user.is_authenticated):
                return False

            org_slug = view.kwargs.get("org_slug")
            if not org_slug:
                return False

            organization, membership = resolve_organization_context(request.user, org_slug)

            if membership is None:
                # Not a real member - but a platform "organizations.manage"
                # holder (Admin/Super Admin) is still granted OWNER-
                # equivalent access to every organization's own pages, the
                # same trust platform_organization_action_view already
                # extends for suspend/delete (see
                # org_permission_service.get_user_org_role's docstring).
                # An unsaved stand-in membership - never queried, never
                # persisted - so every view body's `request.org_membership.role`
                # keeps working exactly as if a real row existed.
                if not user_has_permission(request.user, "organizations.manage"):
                    return False
                membership = OrganizationMembership(organization=organization, user=request.user, role=ORG_ROLE_OWNER)

            if not all(user_has_org_permission(request.user, organization, code) for code in codenames):
                return False

            request.organization = organization
            request.org_membership = membership
            return True

    return _HasOrgPermission


def HasOrgFeatureAccess(feature_code):
    """
    Factory mirroring HasPagePermission, but gating one
    RAG.models.FEATURE_CODES entry via
    org_member_feature_service.has_feature_access() (Plan-level
    inclusion -> per-member override). Resolves the organization via
    resolve_request_organization(request) (the X-Organization-Slug
    header), not the URL's org_slug - every current caller
    (knowledge_views.py, analytics_views.py, reports_views.py,
    ai_tasks_views.py) is a pre-existing shared endpoint reached from
    whichever workspace is active in the sidebar, not a dedicated
    org-scoped URL, the same choice those views already made for their
    own organization resolution.

    Personal Workspace (organization is None) now goes through that
    user's own Personal Plan's included_features via has_feature_access()
    - Personal Workspaces have real Plans too now, see
    org_member_feature_service.has_feature_access()'s docstring.

    Usage: permission_classes = [HasPagePermission("pages.analytics"), HasOrgFeatureAccess("analytics")]
    (add alongside the existing platform-permission check, never in
    place of it).
    """

    class _HasOrgFeatureAccess(BasePermission):
        message = "This feature isn't included in your organization's plan, or has been restricted for your account."

        def has_permission(self, request, view):
            if not (request.user and request.user.is_authenticated):
                return False
            organization, _ = resolve_request_organization(request)
            return has_feature_access(request.user, organization, feature_code)

    return _HasOrgFeatureAccess
