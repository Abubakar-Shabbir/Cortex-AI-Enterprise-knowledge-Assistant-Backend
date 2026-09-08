from django.db import migrations
from django.db.models import OuterRef, Subquery


def backfill_organization(apps, schema_editor):
    AIRequestTrace = apps.get_model("RAG", "AIRequestTrace")
    QueryLog = apps.get_model("RAG", "QueryLog")
    AITaskRun = apps.get_model("RAG", "AITaskRun")

    AIRequestTrace.objects.filter(
        organization__isnull=True, source="ask_ai", query_log__isnull=False
    ).update(
        organization=Subquery(
            QueryLog.objects.filter(pk=OuterRef("query_log_id")).values("organization")[:1]
        )
    )
    AIRequestTrace.objects.filter(
        organization__isnull=True, source="ai_task", ai_task_run__isnull=False
    ).update(
        organization=Subquery(
            AITaskRun.objects.filter(pk=OuterRef("ai_task_run_id")).values("organization")[:1]
        )
    )


def noop_reverse(apps, schema_editor):
    # Irreversible by design - re-deriving which rows were NULL before this
    # migration ran is not recoverable, and rolling back to NULL would just
    # re-lose information a later run of this same migration would redo.
    pass


class Migration(migrations.Migration):

    dependencies = [
        ("RAG", "0050_plan_included_features_and_member_feature_access"),
    ]

    operations = [
        migrations.RunPython(backfill_organization, noop_reverse),
    ]
