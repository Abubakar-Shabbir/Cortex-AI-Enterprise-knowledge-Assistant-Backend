"""
Seed (or reset) the one built-in "Super Admin" account - idempotent,
same get_or_create shape as seed_rbac.py/seed_e2e_user.py, but safe to
run against a real environment (unlike seed_e2e_user.py, this is NOT
DEBUG-gated).

Requires seed_rbac to have already run - this command only assigns the
"super_admin" Role, it never creates or edits its permission set (that
stays seed_rbac.py's sole responsibility, see its module docstring).

Usage:

    python manage.py seed_rbac
    python manage.py seed_super_admin

The password is never hardcoded in source: it's read from the
SUPER_ADMIN_PASSWORD environment variable if set, otherwise a random
one is generated and printed once. Re-running always resets the
password to whatever is resolved this run (env var if set, otherwise a
fresh random one) and force-reassigns the "super_admin" role even if
something else changed it - this is a fixed built-in account, not a
one-time bootstrap.

is_staff/is_superuser are deliberately left False: this account only
ever uses the app's own permission-gated SPA/API surface, never
Django's own /django-admin/ - see RAG/admin.py's RBACAdminOnlyMixin
docstring for why some Django Admin pages aren't safely permission-
scoped the way the SPA is.
"""

import os
import secrets

from django.contrib.auth.models import User
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from RAG.models import Role, SUPER_ADMIN_ROLE_SLUG, UserProfile, UserRole

SUPER_ADMIN_USERNAME = "superadmin"
SUPER_ADMIN_EMAIL = "superadmin@example.invalid"


class Command(BaseCommand):
    help = 'Seed (or reset) the one built-in "Super Admin" account.'

    @transaction.atomic
    def handle(self, *args, **options):
        role = Role.objects.filter(slug=SUPER_ADMIN_ROLE_SLUG).first()
        if role is None:
            raise CommandError(
                'No "super_admin" role found - run `python manage.py seed_rbac` first.'
            )

        password = os.environ.get("SUPER_ADMIN_PASSWORD") or secrets.token_urlsafe(16)

        user, created = User.objects.get_or_create(
            username=SUPER_ADMIN_USERNAME,
            defaults={"email": SUPER_ADMIN_EMAIL, "is_active": True, "is_staff": False, "is_superuser": False},
        )
        user.set_password(password)
        user.email = SUPER_ADMIN_EMAIL
        user.is_active = True
        user.is_staff = False
        user.is_superuser = False
        user.save()

        profile, _ = UserProfile.objects.get_or_create(user=user)
        update_fields = []
        if not profile.email_verified:
            profile.email_verified = True
            update_fields.append("email_verified")
        if profile.account_type != UserProfile.AccountType.PERSONAL:
            profile.account_type = UserProfile.AccountType.PERSONAL
            update_fields.append("account_type")
        if update_fields:
            profile.save(update_fields=update_fields)

        UserRole.objects.update_or_create(user=user, defaults={"role": role})

        self.stdout.write(self.style.SUCCESS(
            f"{'Created' if created else 'Reset'} Super Admin account.\n"
            f"  username: {SUPER_ADMIN_USERNAME}\n"
            f"  password: {password}"
        ))
