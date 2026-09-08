"""
Organization CRUD - creating a new tenant, updating its profile, and
the platform-level (Super Admin, via "organizations.manage" - see
org_permission_service's module docstring) suspend/delete actions.
Member-level mutations (invite/remove/role changes) live in
org_membership_service.py instead - kept separate so "who can create
an organization at all" and "who can manage members within one" stay
two different, independently reasoned-about questions.
"""

import re

from django.db import IntegrityError
from django.utils.text import slugify

from ..models import Organization, OrganizationMembership, UserProfile
from .org_audit_log_service import log_org_activity


class OrganizationServiceError(Exception):
    """Raised for every organization-CRUD failure a caller should show back to the user (duplicate slug, ...) - mirrors org_invitation_service.InvitationError's one-exception-type-per-module shape."""


def _unique_slug(name):
    """
    A URL-safe slug derived from `name`, disambiguated with a numeric
    suffix on collision - the same "try the plain slug, then -2, -3, ..."
    pattern Tag/Category's own slug fields rely on implicitly via their
    (user, slug) uniqueness, made explicit here since Organization.slug
    is globally unique (there's no per-user scope to fall back on).
    """

    base = slugify(name)[:150] or "organization"
    slug = base
    suffix = 2

    while Organization.objects.filter(slug=slug).exists():
        slug = f"{base}-{suffix}"
        suffix += 1

    return slug


def create_organization(name, org_type, created_by, slug=None, **profile_fields):
    """
    Create a new Organization and make `created_by` its Owner in one
    step - an organization with zero members would be an immediate
    lockout (nobody could ever manage it), so this function's contract
    is "always leaves exactly one Owner behind", not "create the org,
    caller adds themselves separately".

    Also the ONE place (alongside org_invitation_service.
    accept_invitation()) that ensures `created_by` ends up
    account_type=COMPANY - every path that ever hands someone their
    first OrganizationMembership must keep that invariant, or
    org_permission_service.resolve_request_organization() (which
    trusts account_type before it even looks at a request header)
    would treat this brand-new Owner as a PERSONAL account with no
    organization access at all, in their own just-created organization.
    """

    slug = slug.strip().lower() if slug else None

    if slug and not re.fullmatch(r"[a-z0-9-]+", slug):
        raise OrganizationServiceError("Slug may only contain lowercase letters, numbers, and hyphens.")

    slug = slug or _unique_slug(name)

    try:
        organization = Organization.objects.create(
            name=name, slug=slug, org_type=org_type, created_by=created_by, **profile_fields,
        )
    except IntegrityError:
        raise OrganizationServiceError(f'The slug "{slug}" is already taken.')

    OrganizationMembership.objects.create(
        organization=organization, user=created_by, role=OrganizationMembership.Role.OWNER,
    )

    profile = created_by.profile
    if profile.account_type != UserProfile.AccountType.COMPANY:
        profile.account_type = UserProfile.AccountType.COMPANY
        profile.save(update_fields=["account_type"])

    log_org_activity(
        organization=organization,
        actor=created_by,
        action="organization.created",
        description=f"{created_by.username} created {organization.name}",
    )

    return organization


def update_organization(organization, actor, request=None, **fields):
    """Apply `fields` (name/description/logo/website/industry/contact_email/contact_phone/org_type - never slug/status, which have their own dedicated, more carefully-guarded paths) to `organization` and log the change. Caller (the view) is responsible for the org.update permission check via org_permission_required."""

    changed = []
    for field, value in fields.items():
        if getattr(organization, field, None) != value:
            setattr(organization, field, value)
            changed.append(field)

    if changed:
        organization.save(update_fields=changed + ["updated_at"])
        log_org_activity(
            organization=organization,
            actor=actor,
            action="organization.updated",
            description=f'{actor.username} updated {", ".join(changed)}',
            metadata={"changed_fields": changed},
            request=request,
        )

    return organization


def suspend_organization(organization, actor, request=None):
    """Platform-level action (organizations.manage) - immediately blocks every member from resolving this organization via resolve_organization_context(), since that function only matches Organization.Status.ACTIVE. Does not delete membership rows, so reactivating restores access exactly as it was."""

    organization.status = Organization.Status.SUSPENDED
    organization.save(update_fields=["status", "updated_at"])
    log_org_activity(
        organization=organization, actor=actor, action="organization.suspended",
        description=f"{actor.username} suspended {organization.name}", request=request,
    )
    return organization


def reactivate_organization(organization, actor, request=None):
    organization.status = Organization.Status.ACTIVE
    organization.save(update_fields=["status", "updated_at"])
    log_org_activity(
        organization=organization, actor=actor, action="organization.reactivated",
        description=f"{actor.username} reactivated {organization.name}", request=request,
    )
    return organization


def delete_organization(organization, actor, request=None):
    """
    Hard delete - CASCADEs to every OrganizationMembership,
    OrganizationInvitation, and (once retrofitted) every organization-
    owned resource, per each model's on_delete=CASCADE. Logs to the
    PLATFORM ActivityLog, not OrganizationAuditLog, since the row this
    function is about to delete is that very audit trail - logging
    "organization deleted" into a table that's about to cease to exist
    would just discard the record of the deletion itself.
    """

    from .activity_log_service import log_activity

    name, org_id = organization.name, organization.id
    organization.delete()

    log_activity(
        actor=actor,
        action="organization.deleted",
        description=f'{actor.username} deleted organization "{name}" (id={org_id})',
        request=request,
    )
