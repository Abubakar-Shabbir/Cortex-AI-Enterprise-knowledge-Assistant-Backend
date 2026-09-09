from django.db import migrations


def backfill_free_plan(apps, schema_editor):
    """
    Every active organization created before organization_service.
    create_organization() started auto-assigning the built-in Free
    plan at creation (billing_service.get_or_create_free_plan()) would
    otherwise be stuck forever at the old "no Subscription row =
    unlimited" default - this makes that guarantee retroactive: no
    active organization should ever be left with no plan at all.

    Deliberately calls the real billing_service functions rather than
    reimplementing Subscription-row creation/credit-refill/period-
    computation here via apps.get_model()'s historical models - this
    is a one-time data correction tightly coupled to right now, not a
    schema change queued for some future release, so the usual
    "migrations must not depend on live app code" caution buys nothing
    and would only risk drifting out of sync with assign_plan()'s own
    (more carefully reasoned) logic. Safe to re-run: assign_plan() is
    itself idempotent per organization (get_or_create on Subscription),
    and the `subscription__isnull=True` filter below only ever touches
    organizations that still have no plan at all.
    """

    from ..models import Organization
    from ..services.billing_service import assign_plan, get_or_create_free_plan

    free_plan = get_or_create_free_plan()
    for organization in Organization.objects.filter(subscription__isnull=True, status="active"):
        assign_plan(organization, free_plan, assigned_by=None)


def noop_reverse(apps, schema_editor):
    # Irreversible by design, same reasoning as 0051's sibling backfill
    # migration - there is no "which organizations had no plan before
    # this ran" to restore, and rolling back to unassigned would just
    # re-create the exact gap this migration exists to close.
    pass


class Migration(migrations.Migration):

    dependencies = [
        ("RAG", "0054_organizationmembership_max_ai_task_runs_per_month_and_more"),
    ]

    operations = [
        migrations.RunPython(backfill_free_plan, noop_reverse),
    ]
