"""
Company Member Registration - 2026-09-06 product decision replacing
the email-invitation-link self-signup flow (org_invitation_service.py,
still present and fully functional, just no longer exposed by the
primary onboarding UI) as how a company gains a new member.

An Owner registers a member by name + email; this module generates a
unique username and a random password, creates (or reactivates) that
person's User account directly - active immediately, no OTP/email-
verification step, since the emailed credentials are themselves the
proof of identity - and emails those credentials exactly once, never
storing the plaintext password anywhere. The member is required to
change that password on first login (see UserProfile.must_change_password
and RAG.api.auth_views.login_view()'s session payload).

A company-registered member has no Personal Workspace at any point:
account_type is set to COMPANY the moment the account exists, unlike
the old accept_invitation() flow, which started a brand-new invitee as
PERSONAL until they actually accepted (see org_invitation_service.py).
This flow is the ONLY thing that creates a company-registered account,
so "only company portal, never Personal Workspace" holds by
construction, not by a later check.
"""

import logging
import secrets
import string

from django.contrib.auth.models import User
from django.utils.text import slugify

from ..models import OrganizationMembership, UserProfile
from .org_audit_log_service import log_org_activity

logger = logging.getLogger(__name__)


class MemberRegistrationError(Exception):
    """Raised for every registration failure a caller should show back to the user - same one-exception-type-per-module shape as InvitationError/MembershipError."""


# Excludes visually-ambiguous characters (0/O, 1/l/I) - this password
# is read off an email and typed once, not saved to a manager, so a
# character a person can't reliably tell apart from another on sight
# is a real support-ticket risk, not just cosmetic.
_PASSWORD_ALPHABET = "abcdefghjkmnpqrstuvwxyzABCDEFGHJKMNPQRSTUVWXYZ23456789!@#$%^&*"
_PASSWORD_LENGTH = 14


def _generate_password():
    return "".join(secrets.choice(_PASSWORD_ALPHABET) for _ in range(_PASSWORD_LENGTH))


def _generate_unique_username(full_name):
    """slugify() a real name into a login-friendly base ('Ada Lovelace' -> 'ada-lovelace'), then append the smallest integer suffix that makes it unique - collisions are expected and normal (two people can share a name), not an error case."""

    base = slugify(full_name)[:25] or "member"
    username = base
    suffix = 1
    while User.objects.filter(username__iexact=username).exists():
        suffix += 1
        username = f"{base}{suffix}"
    return username


def register_company_member(organization, full_name, email, actor, request=None):
    """
    Create (or reactivate) a User for `email`, grant them Member role
    in `organization`, and dispatch their generated credentials by
    email. Returns the (possibly pre-existing, now ACTIVE) membership.

    Three cases:
    - No User with this email exists yet: create one - generated
      username, generated password, active immediately, account_type
      COMPANY, must_change_password set.
    - A User with this email exists but is INACTIVE: this is the
      reversible-removal path org_membership_service.remove_member()'s
      docstring describes - reactivate the SAME account (never create
      a duplicate) with a freshly generated password (the old one was
      never stored in recoverable form, and re-mailing a fresh one
      keeps every registration/re-registration on the same "you always
      get mailed a fresh credential" contract).
    - A User with this email exists and is ACTIVE (a real account
      already in use, anywhere - Personal, another organization, or
      already a member here): refused outright. This flow never
      silently annexes an existing identity into a company.
    """

    full_name = (full_name or "").strip()
    email = (email or "").strip().lower()

    if not full_name:
        raise MemberRegistrationError("A full name is required.")
    if not email:
        raise MemberRegistrationError("An email address is required.")

    if OrganizationMembership.objects.filter(organization=organization, user__email__iexact=email).exists():
        raise MemberRegistrationError("This person is already a member of this organization.")

    # A seat not yet taken isn't a seat used, but registering one
    # always becomes one - checked here, not only after the account is
    # created, so an org can't register its way past its seat limit
    # and only discover the problem after already emailing credentials
    # (same precedent as org_invitation_service.create_invitation()).
    from .billing_service import check_seat_limit
    check_seat_limit(organization)

    existing = User.objects.filter(email__iexact=email).first()
    raw_password = _generate_password()
    reactivated = False

    if existing is None:
        name_parts = full_name.split()
        user = User.objects.create_user(
            username=_generate_unique_username(full_name),
            email=email,
            password=raw_password,
            first_name=name_parts[0],
            last_name=" ".join(name_parts[1:]),
        )
    elif not existing.is_active:
        user = existing
        user.set_password(raw_password)
        user.is_active = True
        user.save(update_fields=["password", "is_active"])
        reactivated = True
    else:
        raise MemberRegistrationError("An active account with this email address already exists.")

    profile = user.profile
    profile.account_type = UserProfile.AccountType.COMPANY
    profile.must_change_password = True
    profile.save(update_fields=["account_type", "must_change_password"])

    membership, created = OrganizationMembership.objects.get_or_create(
        organization=organization, user=user,
        defaults={"role": OrganizationMembership.Role.MEMBER, "invited_by": actor},
    )
    if not created and membership.status != OrganizationMembership.Status.ACTIVE:
        membership.status = OrganizationMembership.Status.ACTIVE
        membership.save(update_fields=["status"])

    log_org_activity(
        organization=organization,
        actor=actor,
        action="member.registered",
        description=(
            f"{actor.username} {'re-registered' if reactivated else 'registered'} "
            f"{user.username} ({email}) as a member"
        ),
        metadata={"target_user_id": user.id, "reactivated": reactivated},
        request=request,
    )

    from . import task_runner
    from ..tasks import send_org_member_credentials_email_task
    task_runner.submit(send_org_member_credentials_email_task, user.id, organization.id, raw_password)

    return membership
