"""
RBAC Decorators

Reusable view-level authorization - every admin-only view wraps
itself in one of these instead of re-checking request.user.is_staff
or duplicating permission logic inline. See RAG/middleware.py for the
matching defense-in-depth layer over the whole /admin/ namespace, and
RAG/services/permission_service.py for the underlying checks.
"""

from functools import wraps

from django.contrib.auth.decorators import login_required
from django.core.exceptions import PermissionDenied

from .services.permission_service import (
    ADMIN,
    has_admin_area_access,
    has_any_settings_permission,
    has_any_system_logs_permission,
    user_has_permission,
    user_has_role,
)
from .services.org_member_feature_service import has_feature_access
from .services.org_permission_service import resolve_organization_context, resolve_request_organization, user_has_org_permission


def role_required(*role_slugs):
    """
    Restrict a view to users whose assigned role slug is one of
    `role_slugs`. Unauthenticated users go through Django's normal
    login flow (login_required); authenticated users with the wrong
    role get a 403 (via Django's standard PermissionDenied handling,
    templates/403.html) rather than a silent redirect that could look
    like the page just doesn't exist.
    """

    def decorator(view_func):
        @wraps(view_func)
        @login_required
        def wrapped_view(request, *args, **kwargs):
            if not user_has_role(request.user, *role_slugs):
                raise PermissionDenied("You don't have access to this page.")
            return view_func(request, *args, **kwargs)
        return wrapped_view
    return decorator


def permission_required(*codenames):
    """
    Restrict a view to users whose role grants every permission in
    `codenames`.
    """

    def decorator(view_func):
        @wraps(view_func)
        @login_required
        def wrapped_view(request, *args, **kwargs):
            if not all(user_has_permission(request.user, code) for code in codenames):
                raise PermissionDenied("You don't have access to this page.")
            return view_func(request, *args, **kwargs)
        return wrapped_view
    return decorator


def org_feature_required(feature_code):
    """
    Plain-Django (non-DRF) counterpart to
    RAG.api.permissions.HasOrgFeatureAccess, for the handful of CSV
    export views that use @permission_required instead of DRF
    permission_classes (RAG/api/reports_views.py's export_*_view
    functions). Resolves the organization via
    resolve_request_organization(request) (the X-Organization-Slug
    header), same as HasOrgFeatureAccess - Personal Workspace always
    passes.
    """

    def decorator(view_func):
        @wraps(view_func)
        @login_required
        def wrapped_view(request, *args, **kwargs):
            organization, _ = resolve_request_organization(request)
            if not has_feature_access(request.user, organization, feature_code):
                raise PermissionDenied(
                    "This feature isn't included in your organization's plan, or has been restricted for your account."
                )
            return view_func(request, *args, **kwargs)
        return wrapped_view
    return decorator


def admin_required(view_func):
    """Shortcut for role_required(ADMIN) - the sole built-in top-tier role now that Super Admin has been removed."""
    return role_required(ADMIN)(view_func)


def admin_area_required(view_func):
    """
    Restrict a view to the Admin role (see
    permission_service.has_admin_area_access - strictly role-based, not
    gated by any individual permission). Used for Admin Overview
    (RAG.views.admin_dashboard_view): the same population that gets the
    admin sidebar shell (context_processors.sidebar_status ->
    can_view_admin_area) must always be able to reach its own Overview
    page too - see get_dashboard_url_for_user for the full rationale.
    Equivalent to admin_required in practice; kept as its own decorator
    so call sites read as "gates the admin area" rather than "gates the
    Admin role" even though today those are the same check.
    """

    @wraps(view_func)
    @login_required
    def wrapped_view(request, *args, **kwargs):
        if not has_admin_area_access(request.user):
            raise PermissionDenied("You don't have access to this page.")
        return view_func(request, *args, **kwargs)
    return wrapped_view


def org_permission_required(*codenames):
    """
    Restrict an organization-scoped view (one taking `org_slug` as a
    URL kwarg) to users who are an ACTIVE member of that organization
    and whose role grants every codename in `codenames` -
    RAG.services.org_permission_service's counterpart to
    permission_required() above, tenant-scoped instead of
    platform-scoped.

    Never trusts org_slug beyond "which organization is being asked
    about": resolve_organization_context() re-derives membership from
    OrganizationMembership on every request. A 404 (org_slug doesn't
    resolve to an active organization) and a 403 (resolves, but this
    user isn't an active member, or is a member without the required
    permission) are kept distinct so a request never learns whether an
    organization it can't access even exists versus genuinely doesn't.

    On success, attaches `request.organization` and
    `request.org_membership` so the view body never has to re-resolve
    either - and, critically, so the view's own queryset filtering has
    a trusted `request.organization` to scope by instead of re-reading
    org_slug itself.
    """

    def decorator(view_func):
        @wraps(view_func)
        @login_required
        def wrapped_view(request, org_slug, *args, **kwargs):
            organization, membership = resolve_organization_context(request.user, org_slug)

            if not membership:
                raise PermissionDenied("You don't have access to this organization.")

            if not all(user_has_org_permission(request.user, organization, code) for code in codenames):
                raise PermissionDenied("You don't have access to this resource.")

            request.organization = organization
            request.org_membership = membership
            return view_func(request, org_slug, *args, **kwargs)
        return wrapped_view
    return decorator


def settings_access_required(view_func):
    """
    Restrict admin_settings_view to any role holding at least one
    settings.manage_* permission - broader than a single specific
    permission on purpose, mirroring admin_area_required's "coarse view
    gate" pattern. Which cards the request actually sees/can edit is
    still scoped per field-group permission inside the view/template
    itself (system_config_service.SETTINGS_PAGE_PERMISSIONS) - this
    decorator only answers "can this role open the page at all."
    """

    @wraps(view_func)
    @login_required
    def wrapped_view(request, *args, **kwargs):
        if not has_any_settings_permission(request.user):
            raise PermissionDenied("You don't have access to this page.")
        return view_func(request, *args, **kwargs)
    return wrapped_view


def system_logs_access_required(view_func):
    """
    Restrict admin_system_logs_view to any role holding at least one of
    permission_service.SYSTEM_LOGS_PERMISSIONS - the consolidated
    System Logs page (Request Traces + Error Groups + Activity tabs)
    behind one nav entry/one URL. Which tabs a given request actually
    sees is scoped per-permission inside the view/template, mirroring
    settings_access_required's "coarse gate + fine-grained internal
    scoping" pattern.
    """

    @wraps(view_func)
    @login_required
    def wrapped_view(request, *args, **kwargs):
        if not has_any_system_logs_permission(request.user):
            raise PermissionDenied("You don't have access to this page.")
        return view_func(request, *args, **kwargs)
    return wrapped_view
