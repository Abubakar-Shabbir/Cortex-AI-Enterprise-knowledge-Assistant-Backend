"""
Seed the default, extensible OrganizationType catalog. Idempotent -
safe to re-run any time (e.g. after adding a new entry to
DEFAULT_ORGANIZATION_TYPES). Mirrors seed_rbac.py's get_or_create
shape: re-running never overwrites an existing type's live name/
is_active state, only fills in whichever ones are still missing.

Usage, after migrating the RAG models:

    python manage.py migrate RAG
    python manage.py seed_organization_types
"""

from django.core.management.base import BaseCommand
from django.db import transaction

from RAG.models import OrganizationType

# (slug, name) - a starting catalog, not an exhaustive one. Admins can
# add further types via Django Admin without any code change, since
# OrganizationType is a plain table, not a hardcoded choices list.
DEFAULT_ORGANIZATION_TYPES = [
    ("company", "Company"),
    ("startup", "Startup"),
    ("enterprise", "Enterprise"),
    ("university", "University"),
    ("government", "Government Organization"),
    ("nonprofit", "Non-Profit"),
    ("agency", "Agency"),
    ("other", "Other"),
]


class Command(BaseCommand):
    help = "Seed the default OrganizationType catalog."

    @transaction.atomic
    def handle(self, *args, **options):

        created_count = 0

        for slug, name in DEFAULT_ORGANIZATION_TYPES:
            org_type, created = OrganizationType.objects.get_or_create(
                slug=slug, defaults={"name": name},
            )
            if created:
                created_count += 1
            self.stdout.write(f"{'Created' if created else 'Exists'} organization type: {slug}")

        self.stdout.write(self.style.SUCCESS(
            f"Organization type seed complete. Created {created_count} new type(s)."
        ))
