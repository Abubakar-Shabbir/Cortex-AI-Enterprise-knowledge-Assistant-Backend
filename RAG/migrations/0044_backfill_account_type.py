# Generated manually - see RAG/migrations/0034_backfill_email_verified_existing_users.py for the same pattern.

from django.db import migrations


def backfill_account_type(apps, schema_editor):
    """
    account_type was introduced after this app already had real users
    and organizations (see 0043) - every account that already holds at
    least one OrganizationMembership (owner or invited member, any
    status) is backfilled to COMPANY; everyone else defaults to
    PERSONAL (the field's own model default, already applied to every
    row by 0043's AddField). Uses the historical model (apps.get_model),
    not the "live" RAG.models.UserProfile.
    """

    UserProfile = apps.get_model("RAG", "UserProfile")
    OrganizationMembership = apps.get_model("RAG", "OrganizationMembership")

    member_user_ids = OrganizationMembership.objects.values_list("user_id", flat=True).distinct()
    UserProfile.objects.filter(user_id__in=member_user_ids).update(account_type="company")


def noop_reverse(apps, schema_editor):
    """Deliberately a no-op, not "set everyone back to personal" - reversing this migration should not retroactively change how existing company accounts are routed."""


class Migration(migrations.Migration):

    dependencies = [
        ('RAG', '0043_organization_size_userprofile_account_type'),
    ]

    operations = [
        migrations.RunPython(backfill_account_type, noop_reverse),
    ]
