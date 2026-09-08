"""
Auth endpoints for the React SPA (frontend/) - thin JSON wrappers
around the exact same logic RAG/auth_views.py's classic login_user()/
logout_user()/signup()/verify_otp()/resend_otp()/RAGPasswordResetView/
RAGPasswordResetConfirmView already run (form classes, rate limiting,
remember-me session expiry, activity logging, the indistinguishable-
from-wrong-password OTP-pending branch, enumeration-resistant password
reset). No authentication/authorization behavior is duplicated or
reimplemented here, only re-exposed as JSON instead of an HTML
redirect/render - the classic Django pages/URLs keep working unchanged
for anyone who reaches them directly (e.g. a stale bookmark).
"""

from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import SetPasswordForm
from django.contrib.auth.tokens import default_token_generator
from django.conf import settings
from django.middleware.csrf import get_token
from django.utils.encoding import force_str
from django.utils.http import urlsafe_base64_decode
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

from ..auth_views import RAGPasswordResetForm, SignupForm, PASSWORD_RESET_ATTEMPTS_PER_EMAIL, PASSWORD_RESET_ATTEMPTS_PER_IP, SIGNUP_ATTEMPTS_PER_IP
from ..models import OrganizationType, Role, User, UserProfile, UserRole
from ..services import otp_service
from ..services.activity_log_service import log_activity
from ..services.geolocation_service import get_client_ip
from ..services.notification_service import create_notification
from ..services.organization_service import OrganizationServiceError, create_organization
from ..services.permission_service import USER, get_user_access_snapshot, has_admin_area_access
from ..services.rate_limit_service import is_rate_limited
from ..utils.formatting import mask_email

LOGIN_ATTEMPTS_PER_USERNAME = (5, 900)
LOGIN_ATTEMPTS_PER_IP = (20, 900)


def _form_errors(form):
    """{field: [messages]} + a "__all__" bucket for non-field errors - same shape for every form-backed endpoint below."""
    errors = {field: messages for field, messages in form.errors.items()}
    return errors


def _resolve_portal(can_view_admin_area, account_type):
    """
    The single backend-computed source of truth for which of the
    three portals (Personal/Company/Platform Admin) the frontend
    should render - never derived independently client-side from
    `role`/`accountType`, so there's exactly one place "which portal"
    is decided. Platform admin access always wins (an Admin/Super
    Admin who also happens to own a company still lands in the
    Platform portal by default - they can still reach their own
    organization's pages directly, this only decides the default
    shell/nav).
    """
    if can_view_admin_area:
        return "platform_admin"
    if account_type == UserProfile.AccountType.COMPANY:
        return "company"
    return "personal"


def _session_payload(request):
    role, can_view_admin_area, user_permissions = get_user_access_snapshot(request.user)
    account_type = getattr(getattr(request.user, "profile", None), "account_type", UserProfile.AccountType.PERSONAL)
    return {
        "authenticated": True,
        "user": {
            "id": request.user.id,
            "username": request.user.username,
            "first_name": request.user.first_name,
            "last_name": request.user.last_name,
            "email": request.user.email,
        },
        "role": role.name if role else None,
        "can_view_admin_area": can_view_admin_area,
        "permissions": user_permissions,
        # Decided once at signup (or by accepting an invitation) - the
        # frontend reads this to decide whether to show the Company
        # workspace switcher at all (never a Personal<->Company choice -
        # see UserProfile.account_type's help_text), but it's read-only
        # information from the frontend's point of view: every actual
        # authorization decision is still made server-side from this
        # same field plus real OrganizationMembership rows, regardless
        # of what the frontend does with it.
        "account_type": account_type,
        "portal": _resolve_portal(can_view_admin_area, account_type),
        # Set by org_member_registration_service.register_company_member()
        # for a company-registered member's generated password - the SPA
        # blocks every route except the password-change page until the
        # user clears this (RAG.api.profile_views.profile_password_view
        # clears it as a side effect of a successful change). Always
        # False for a self-service Personal/Company signup.
        "must_change_password": getattr(getattr(request.user, "profile", None), "must_change_password", False),
        "csrf_token": get_token(request),
    }


@api_view(["GET"])
@permission_classes([AllowAny])
def session_view(request):
    """
    Bootstrap endpoint the SPA calls once on load - also the only
    reliable place to hand the CSRF token to JS (see api/permissions.py
    module docstring / settings.py's REST_FRAMEWORK comment on why the
    cookie itself is httponly). Anonymous callers still get a fresh
    token back (get_token() sets the cookie as a side effect) so the
    very first login POST already carries a valid one.
    """

    if not request.user.is_authenticated:
        return Response({"authenticated": False, "csrf_token": get_token(request)})

    return Response(_session_payload(request))


@api_view(["POST"])
@permission_classes([AllowAny])
def login_view(request):
    username = (request.data.get("username") or "").strip()
    password = request.data.get("password") or ""
    remember_me = bool(request.data.get("remember_me"))

    ip = get_client_ip(request)

    username_limited = username and is_rate_limited(f"login:user:{username.lower()}", *LOGIN_ATTEMPTS_PER_USERNAME)
    ip_limited = ip and is_rate_limited(f"login:ip:{ip}", *LOGIN_ATTEMPTS_PER_IP)

    if username_limited or ip_limited:
        return Response(
            {"error": "Too many login attempts. Please wait a few minutes and try again."},
            status=429,
        )

    user = authenticate(request, username=username, password=password)

    if user is None and username and password:
        candidate = User.objects.filter(username__iexact=username, is_active=False).first()
        if candidate and candidate.check_password(password) and otp_service.has_pending_verification(candidate):
            request.session["pending_verification_user_id"] = candidate.id
            return Response({"pending_verification": True, "redirect": "/verify-otp/"})

    if user is None:
        return Response({"error": "Invalid username or password."}, status=400)

    login(request, user)

    if remember_me:
        request.session.set_expiry(settings.REMEMBER_ME_SESSION_AGE)
    else:
        request.session.set_expiry(0)

    log_activity(
        actor=user,
        action="user.login",
        description=f"{user.username} logged in",
        request=request,
    )

    return Response(_session_payload(request))


@api_view(["POST"])
def logout_view(request):
    log_activity(
        actor=request.user,
        action="user.logout",
        description=f"{request.user.username} logged out",
        request=request,
    )
    logout(request)
    return Response({"authenticated": False, "csrf_token": get_token(request)})


@api_view(["POST"])
@permission_classes([AllowAny])
def signup_view(request):
    """
    JSON wrapper around RAG.auth_views.signup()'s POST branch - same
    SignupForm, same rate limit, same inactive-user-plus-OTP account
    creation - plus the Personal vs Company decision this account
    starts with.

    `account_type` ("personal" or "company", default "personal" so an
    invitation-flow signup that never shows this choice at all still
    works) is a one-time decision, not something a workspace switcher
    ever changes afterward - see UserProfile.account_type's help_text.
    A "company" signup also registers the Organization and makes this
    user its Owner, atomically with the account itself; an invited
    employee instead stays "personal" here and is flipped to "company"
    only once they actually accept that invitation (see
    org_invitation_service.accept_invitation()) - signing up alone,
    without accepting anything, must never grant organization access.
    """

    ip = get_client_ip(request)
    if ip and is_rate_limited(f"signup:ip:{ip}", *SIGNUP_ATTEMPTS_PER_IP):
        return Response({"error": "Too many signup attempts from this location. Please try again later."}, status=429)

    account_type = (request.data.get("account_type") or UserProfile.AccountType.PERSONAL).strip().lower()
    if account_type not in (UserProfile.AccountType.PERSONAL, UserProfile.AccountType.COMPANY):
        return Response({"errors": {"account_type": ["Choose Personal or Company."]}}, status=400)

    company_fields = {}
    company_org_type = None
    if account_type == UserProfile.AccountType.COMPANY:
        company_name = (request.data.get("company_name") or "").strip()
        if not company_name:
            return Response({"errors": {"company_name": ["Company name is required."]}}, status=400)

        company_org_type = OrganizationType.objects.filter(
            slug=(request.data.get("company_org_type") or "").strip(), is_active=True,
        ).first()
        if not company_org_type:
            return Response({"errors": {"company_org_type": ["Choose a valid organization type."]}}, status=400)

        company_fields = {
            "description": (request.data.get("company_description") or "").strip(),
            "website": (request.data.get("company_website") or "").strip(),
            "industry": (request.data.get("company_industry") or "").strip(),
            "size": (request.data.get("company_size") or "").strip(),
            "contact_email": (request.data.get("company_contact_email") or "").strip(),
            "contact_phone": (request.data.get("company_contact_phone") or "").strip(),
        }

    form = SignupForm(request.data)

    if not form.is_valid():
        return Response({"errors": _form_errors(form)}, status=400)

    name_parts = form.cleaned_data["full_name"].split()
    first_name = name_parts[0] if name_parts else ""
    last_name = " ".join(name_parts[1:])

    new_user = User.objects.create_user(
        username=form.cleaned_data["username"],
        email=form.cleaned_data["email"],
        password=form.cleaned_data["password"],
        first_name=first_name,
        last_name=last_name,
        is_active=False,
    )

    default_role, _ = Role.objects.get_or_create(slug=USER, defaults={"name": "User", "is_system": True})
    UserRole.objects.create(user=new_user, role=default_role)

    new_user.profile.account_type = account_type
    new_user.profile.save(update_fields=["account_type"])

    if account_type == UserProfile.AccountType.COMPANY:
        try:
            create_organization(
                name=company_name,
                org_type=company_org_type,
                created_by=new_user,
                slug=request.data.get("company_slug"),
                **company_fields,
            )
        except OrganizationServiceError as e:
            new_user.delete()
            return Response({"errors": {"company_name": [str(e)]}}, status=400)

    log_activity(actor=new_user, action="user.signed_up", description=f"{new_user.username} created an account", request=request)

    otp_service.generate_and_send_otp(new_user)
    request.session["pending_verification_user_id"] = new_user.id

    return Response({"masked_email": mask_email(new_user.email)})


def _pending_verification_user(request):
    user_id = request.session.get("pending_verification_user_id")
    user = User.objects.filter(id=user_id).first() if user_id else None
    return user if user and not user.is_active else None


@api_view(["GET"])
@permission_classes([AllowAny])
def verify_otp_status_view(request):
    """Backs the /verify-otp/ SPA page's initial render - who (if anyone) is pending, so a direct/refreshed visit knows whether to show the code form or bounce to signup."""

    user = _pending_verification_user(request)
    if user is None:
        return Response({"pending": False})
    return Response({"pending": True, "masked_email": mask_email(user.email)})


@api_view(["POST"])
@permission_classes([AllowAny])
def verify_otp_view(request):
    """JSON wrapper around RAG.auth_views.verify_otp()'s POST branch - on success, logs the user in exactly like the classic view (same session behavior) and returns the same session payload login_view returns."""

    user = _pending_verification_user(request)
    if user is None:
        return Response({"error": "No pending verification.", "pending": False}, status=400)

    code = (request.data.get("code") or "").strip()
    success, status = otp_service.verify_otp(user, code)

    if not success:
        error_messages = {
            "expired": "This code has expired. Request a new one below.",
            "invalid": "That code isn't right. Please try again.",
            "max_attempts": "Too many incorrect attempts. Request a new code below.",
            "none_pending": "No active code found. Request a new one below.",
            "rate_limited": "Too many attempts. Please wait a few minutes and try again.",
        }
        return Response({"error": error_messages.get(status, "Something went wrong. Please try again.")}, status=400)

    del request.session["pending_verification_user_id"]

    login(request, user)
    request.session.set_expiry(0)

    log_activity(actor=user, action="user.email_verified", description=f"{user.username} verified their email", request=request)

    return Response(_session_payload(request))


@api_view(["POST"])
@permission_classes([AllowAny])
def resend_otp_view(request):
    """JSON wrapper around RAG.auth_views.resend_otp() - identical cooldown/hourly-limit contract."""

    user = _pending_verification_user(request)
    if user is None:
        return Response({"error": "No pending verification."}, status=400)

    allowed, seconds_remaining = otp_service.can_resend(user)
    if not allowed:
        return Response({"error": "Please wait before requesting another code.", "cooldown_seconds": seconds_remaining}, status=429)

    otp_service.generate_and_send_otp(user)
    return Response({"cooldown_seconds": otp_service.RESEND_COOLDOWN_SECONDS})


@api_view(["POST"])
@permission_classes([AllowAny])
def password_reset_request_view(request):
    """
    JSON wrapper around RAG.auth_views.RAGPasswordResetView - same
    form/rate limits/enumeration-resistance (always looks identical to
    the caller whether or not the email exists or was rate-limited).
    """

    email = (request.data.get("email") or "").strip().lower()
    ip = get_client_ip(request)

    form = RAGPasswordResetForm(request.data)
    if not form.is_valid():
        return Response({"errors": _form_errors(form)}, status=400)

    email_limited = email and is_rate_limited(f"pwreset:{email}", *PASSWORD_RESET_ATTEMPTS_PER_EMAIL)
    ip_limited = ip and is_rate_limited(f"pwreset:ip:{ip}", *PASSWORD_RESET_ATTEMPTS_PER_IP)

    if not email_limited and not ip_limited:
        form.save(request=request)

    # Same response either way (sent or silently rate-limited) - matches
    # RAGPasswordResetView.form_valid()'s enumeration-resistant redirect.
    return Response({"ok": True})


def _user_from_uidb64(uidb64):
    try:
        return User.objects.get(pk=force_str(urlsafe_base64_decode(uidb64)))
    except (TypeError, ValueError, OverflowError, User.DoesNotExist):
        return None


@api_view(["GET"])
@permission_classes([AllowAny])
def password_reset_validate_view(request, uidb64, token):
    """Backs PasswordResetConfirm.jsx's initial render - whether the emailed link is still good, same check password_reset_confirm.html's `validlink` reflects."""

    user = _user_from_uidb64(uidb64)
    valid = bool(user and default_token_generator.check_token(user, token))
    return Response({"valid": valid})


@api_view(["POST"])
@permission_classes([AllowAny])
def password_reset_confirm_view(request, uidb64, token):
    """JSON wrapper around RAG.auth_views.RAGPasswordResetConfirmView.form_valid() - same SetPasswordForm (so the same AUTH_PASSWORD_VALIDATORS enforcement), same post-reset activity log + notification."""

    user = _user_from_uidb64(uidb64)
    if user is None or not default_token_generator.check_token(user, token):
        return Response({"error": "This link is invalid or has expired."}, status=400)

    form = SetPasswordForm(user, request.data)
    if not form.is_valid():
        return Response({"errors": _form_errors(form)}, status=400)

    form.save()

    log_activity(actor=user, action="user.password_reset", description=f"{user.username} reset their password via email", request=request)
    create_notification(
        recipient=user,
        notification_type="account.password_changed",
        title="Your password was changed",
        message="Your password was just reset. If this wasn't you, contact support immediately.",
    )

    return Response({"ok": True})
