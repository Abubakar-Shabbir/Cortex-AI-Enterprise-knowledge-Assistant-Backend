"""
Organization Invitations - create/accept/revoke, with the security
properties a join-by-token flow needs: an opaque, unguessable,
single-use token; a hard expiry; and acceptance bound to the invited
email address specifically, not "whoever holds the token" alone.

Deliberately does not send the actual email itself - create_invitation()
only creates the row and returns it; the caller (organizations_views.
organization_invitations_view) dispatches RAG.tasks.
send_org_invitation_email_task via task_runner.submit() right after,
the same "service does the mutation, the view dispatches the
background side effect" split RAG/views.py already uses for
send_share_invite_email_task. Keeping email dispatch out of this
module means every other caller (tests, a future CLI/admin action)
gets a plain, side-effect-free invitation row without also triggering
an email as an unavoidable side effect.
"""

import logging
import secrets
from datetime import timedelta

from django.db import IntegrityError
from django.utils import timezone

from ..models import Organization, OrganizationInvitation, OrganizationMembership, UserProfile
from .org_audit_log_service import log_org_activity

logger = logging.getLogger(__name__)

INVITATION_EXPIRY_DAYS = 7

# len() of secrets.token_urlsafe(32)'s output (~43 chars) comfortably
# fits the model's max_length=64 with room to spare if the generation
# scheme ever changes to a slightly longer token.
TOKEN_BYTES = 32


class InvitationError(Exception):
    """Raised for every invitation-flow failure the caller is expected to show back to a user (already-pending invite, expired/revoked/wrong-email token, ...) - never a bare ValueError, so views can catch this one type and turn it into a clean 400 response."""


def create_invitation(organization, email, role, invited_by, request=None):
    """
    Create a PENDING invitation. Idempotent for an already-pending
    invite to the same (organization, email): rather than raising or
    silently creating a duplicate that would collide with the DB's own
    partial unique constraint, returns the existing PENDING row
    unchanged - re-clicking "Invite" for someone already invited is a
    no-op, not an error.
    """

    email = email.strip().lower()

    existing = OrganizationInvitation.objects.filter(
        organization=organization, email=email, status=OrganizationInvitation.Status.PENDING,
    ).first()

    if existing:
        return existing

    # A pending invitation isn't a seat yet, but accepting one always
    # becomes one - checked here, not only at seat-count time on
    # acceptance, so an org can't invite its way past its seat limit
    # and only discover the problem when someone tries to accept.
    from .billing_service import check_seat_limit
    check_seat_limit(organization)

    token = secrets.token_urlsafe(TOKEN_BYTES)

    try:
        invitation = OrganizationInvitation.objects.create(
            organization=organization,
            email=email,
            role=role,
            token=token,
            invited_by=invited_by,
            expires_at=timezone.now() + timedelta(days=INVITATION_EXPIRY_DAYS),
        )
    except IntegrityError:
        # Lost a race against a concurrent create_invitation() call for
        # the same (organization, email) - the DB's partial unique
        # constraint is the actual source of truth here, this is just
        # a friendlier response than a raw 500 for that narrow window.
        existing = OrganizationInvitation.objects.filter(
            organization=organization, email=email, status=OrganizationInvitation.Status.PENDING,
        ).first()
        if existing:
            return existing
        raise

    log_org_activity(
        organization=organization,
        actor=invited_by,
        action="member.invited",
        description=f'{invited_by.username if invited_by else "Someone"} invited {email} as {role}',
        metadata={"email": email, "role": role, "invitation_id": invitation.id},
        request=request,
    )

    return invitation


def _mark_expired_if_due(invitation):
    """
    Lazily flips a PENDING invitation to EXPIRED the moment it's
    looked at past its expiry, rather than requiring a periodic sweep
    job - status is only ever read through get_invitation_by_token()/
    accept_invitation() below, so this guarantees an expired token can
    never be accepted regardless of whether any cleanup task has run.
    """

    if invitation.status == OrganizationInvitation.Status.PENDING and invitation.expires_at <= timezone.now():
        invitation.status = OrganizationInvitation.Status.EXPIRED
        invitation.save(update_fields=["status"])

    return invitation


def get_invitation_by_token(token):
    """The invitation for `token`, with lazy expiry applied, or None for an unknown token. Never raises - a bad/forged token is just "not found", the same fail-closed shape the rest of this codebase's security-sensitive lookups use."""

    invitation = OrganizationInvitation.objects.select_related("organization").filter(token=token).first()

    if invitation:
        invitation = _mark_expired_if_due(invitation)

    return invitation


def accept_invitation(token, user, request=None):
    """
    Accept a PENDING, unexpired invitation as `user`. Requires
    `user.email` to case-insensitively match the invitation's email -
    the binding that stops a token from being redeemed into an account
    it wasn't actually issued to, even if the token string itself
    leaked to someone else. Idempotent for a user who's somehow already
    a member (get_or_create rather than create): their existing role
    is left untouched rather than silently re-granted/changed by
    replaying an old invite.

    Raises InvitationError (never a bare exception) for every failure
    case - unknown token, wrong status, expired, email mismatch - so
    callers can show one consistent "this invitation isn't valid"
    message without inspecting exception internals.
    """

    invitation = get_invitation_by_token(token)

    if not invitation:
        raise InvitationError("This invitation link is invalid.")

    if invitation.status != OrganizationInvitation.Status.PENDING:
        raise InvitationError(f"This invitation is {invitation.get_status_display().lower()} and can no longer be accepted.")

    if not user.email or user.email.strip().lower() != invitation.email:
        raise InvitationError("This invitation was sent to a different email address than your account's.")

    membership, created = OrganizationMembership.objects.get_or_create(
        organization=invitation.organization,
        user=user,
        defaults={"role": invitation.role, "invited_by": invitation.invited_by},
    )

    # Personal vs Company is decided once, at signup - but a brand-new
    # invited employee (Personal vs Company spec, section 5-6) signed
    # up without ever seeing that choice, so they still default to
    # PERSONAL until this exact moment. Accepting a real invitation is
    # what actually makes them a company user; an existing PERSONAL
    # account that later accepts an invitation converts the same way -
    # a COMPANY account's requests are never scoped to Personal
    # Workspace (org_permission_service.resolve_request_organization()),
    # so holding any OrganizationMembership at all means this must be
    # "company" going forward, not a mix of both.
    profile = user.profile
    if profile.account_type != UserProfile.AccountType.COMPANY:
        profile.account_type = UserProfile.AccountType.COMPANY
        profile.save(update_fields=["account_type"])

    invitation.status = OrganizationInvitation.Status.ACCEPTED
    invitation.accepted_by = user
    invitation.accepted_at = timezone.now()
    invitation.save(update_fields=["status", "accepted_by", "accepted_at"])

    log_org_activity(
        organization=invitation.organization,
        actor=user,
        action="member.joined",
        description=f"{user.username} accepted an invitation and joined as {membership.role}",
        metadata={"invitation_id": invitation.id, "already_member": not created},
        request=request,
    )

    return membership


def revoke_invitation(invitation, actor, request=None):
    """
    Revoke a PENDING invitation - the token becomes permanently
    unusable (status only ever moves PENDING -> {ACCEPTED, EXPIRED,
    REVOKED}, never back). A no-op (returns False) for an invitation
    that's already left the PENDING state, rather than an error -
    revoking twice, or revoking an already-accepted invite, has
    nothing further to do.

    The PENDING check and the write happen as ONE atomic conditional
    UPDATE (.filter(status=PENDING).update(...)), not a read-then-write
    against `invitation`'s possibly-stale in-memory .status - a caller
    can hold a fetched invitation object from moments (or a concurrent
    request) earlier, and trusting that in-memory value instead of the
    database's current row would let a revoke silently overwrite an
    invitation accept_invitation() already accepted out from under it,
    corrupting a real membership-granting event's status.
    """

    updated = OrganizationInvitation.objects.filter(
        pk=invitation.pk, status=OrganizationInvitation.Status.PENDING,
    ).update(status=OrganizationInvitation.Status.REVOKED, revoked_at=timezone.now())

    if not updated:
        return False

    invitation.refresh_from_db()

    log_org_activity(
        organization=invitation.organization,
        actor=actor,
        action="member.invitation_revoked",
        description=f'{actor.username if actor else "Someone"} revoked the invitation for {invitation.email}',
        metadata={"invitation_id": invitation.id, "email": invitation.email},
        request=request,
    )

    return True


def list_invitations(organization, status=None):
    """Every invitation for `organization`, newest first, optionally filtered to one status - powers the Members page's pending-invitations list."""

    qs = OrganizationInvitation.objects.filter(organization=organization)
    if status:
        qs = qs.filter(status=status)
    return qs.order_by("-created_at")
