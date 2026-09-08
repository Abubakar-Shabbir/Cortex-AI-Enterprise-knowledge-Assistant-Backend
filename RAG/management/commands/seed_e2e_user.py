"""
Seed (or reset) the disposable accounts Playwright end-to-end tests
log in as - idempotent, safe to re-run any time, same get_or_create
shape as seed_rbac.py/seed_organization_types.py.

Personal vs Company is a one-time signup decision (see UserProfile.
account_type's help_text) - one seeded account can never legitimately
cover both, so this seeds TWO:

- e2e_personal_user: account_type=PERSONAL, zero organizations, ever.
- e2e_company_user: account_type=COMPANY, Owner of exactly one
  organization ("E2E Test Company") to start from - the Playwright
  suite's own organizations.spec.js exercises registering a SECOND
  company from there (self-service creation is still available to an
  existing Company account - see organizations_views.organizations_view).

Deliberately dedicated, obviously-named accounts rather than reusing
any real user already in this dev database - E2E tests create/mutate
real rows (documents, organizations, memberships) against whichever
database `manage.py runserver` is pointed at, and must never risk
touching an actual person's account or data.

Usage, against a running dev database:

    python manage.py seed_e2e_user

The password is intentionally hardcoded (not read from the
environment) - this command must never be run against a production
database; it exists purely to give the Playwright suite (frontend/e2e/)
a stable, known-credentials account in a local dev environment.
"""

from django.conf import settings
from django.contrib.auth.models import User
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from RAG.models import USER_ROLE_SLUG, Organization, OrganizationMembership, OrganizationType, Role, UserProfile, UserRole

E2E_PASSWORD = "E2ePlaywright!2026"

E2E_PERSONAL_USERNAME = "e2e_personal_user"
E2E_PERSONAL_EMAIL = "e2e_personal_user@example.invalid"

E2E_COMPANY_USERNAME = "e2e_company_user"
E2E_COMPANY_EMAIL = "e2e_company_user@example.invalid"
E2E_COMPANY_NAME = "E2E Test Company"
E2E_COMPANY_SLUG = "e2e-test-company"


class Command(BaseCommand):
    help = "Seed the disposable e2e_personal_user/e2e_company_user accounts Playwright logs in as."

    @transaction.atomic
    def handle(self, *args, **options):
        if not settings.DEBUG:
            raise CommandError(
                "Refusing to run seed_e2e_user with DEBUG=False - this command creates "
                "known-password test accounts and must only ever run against a local dev database."
            )

        default_role = Role.objects.filter(slug=USER_ROLE_SLUG).first()

        def _seed_base_user(username, email, account_type):
            user, created = User.objects.get_or_create(username=username, defaults={"email": email, "is_active": True})
            user.set_password(E2E_PASSWORD)
            user.email = email
            user.is_active = True
            user.save()

            profile, _ = UserProfile.objects.get_or_create(user=user)
            update_fields = []
            if not profile.email_verified:
                profile.email_verified = True
                update_fields.append("email_verified")
            if profile.account_type != account_type:
                profile.account_type = account_type
                update_fields.append("account_type")
            if update_fields:
                profile.save(update_fields=update_fields)

            if default_role and not UserRole.objects.filter(user=user).exists():
                UserRole.objects.create(user=user, role=default_role)

            return user, created

        personal_user, personal_created = _seed_base_user(
            E2E_PERSONAL_USERNAME, E2E_PERSONAL_EMAIL, UserProfile.AccountType.PERSONAL,
        )
        # A PERSONAL account must never hold organization membership -
        # if a previous run (before this account_type split existed)
        # left any behind, drop them so this account stays a clean
        # Personal-only fixture.
        OrganizationMembership.objects.filter(user=personal_user).delete()

        company_user, company_created = _seed_base_user(
            E2E_COMPANY_USERNAME, E2E_COMPANY_EMAIL, UserProfile.AccountType.COMPANY,
        )

        org_type, _ = OrganizationType.objects.get_or_create(slug="company", defaults={"name": "Company"})
        organization, org_created = Organization.objects.get_or_create(
            slug=E2E_COMPANY_SLUG, defaults={"name": E2E_COMPANY_NAME, "org_type": org_type, "created_by": company_user},
        )
        OrganizationMembership.objects.get_or_create(
            organization=organization, user=company_user,
            defaults={"role": OrganizationMembership.Role.OWNER},
        )

        self.stdout.write(self.style.SUCCESS(
            f"{'Created' if personal_created else 'Reset'} {E2E_PERSONAL_USERNAME} (personal, password: {E2E_PASSWORD})\n"
            f"{'Created' if company_created else 'Reset'} {E2E_COMPANY_USERNAME} (company, owner of '{E2E_COMPANY_NAME}', password: {E2E_PASSWORD})"
        ))
