"""
Seed the built-in Free company plan. Idempotent - safe to re-run any
time; only fills in the plan if it doesn't already exist (an admin may
have since edited its live settings, which this must never overwrite).

organization_service.create_organization() also calls the same
billing_service.get_or_create_free_plan() directly, so a fresh
environment gets a working Free plan on the very first company signup
even if this command was never run - this command exists mainly so the
Free plan shows up in Admin > Billing Plans (and is inspectable/
editable there) before anyone has signed up yet, matching the other
seed_* commands' role of pre-populating reference data rather than
leaving it to be lazily created on first use.

Usage, after migrating the RAG models:

    python manage.py migrate RAG
    python manage.py seed_billing_plans
"""

from django.core.management.base import BaseCommand
from django.db import transaction

from RAG.services.billing_service import get_or_create_free_plan


class Command(BaseCommand):
    help = "Seed the built-in Free company plan."

    @transaction.atomic
    def handle(self, *args, **options):
        plan = get_or_create_free_plan()
        self.stdout.write(self.style.SUCCESS(f'Free plan ready: "{plan.name}" (slug="{plan.slug}").'))
