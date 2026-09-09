"""
CSV report export (Reports page).

Uses the stdlib `csv` module writing straight into the HttpResponse
the view returns - no new dependency, no intermediate file on disk.
Each function here only builds the *rows*; the view is responsible
for the HttpResponse/csv.writer plumbing, so these stay easy to unit
test without touching the request/response cycle.
"""

from ..models import AITaskRun, Document, QueryLog

DOCUMENTS_REPORT_HEADER = [
    "Title", "File Type", "File Size (bytes)", "Chunk Count", "Uploaded At",
]

AI_TASK_RUNS_REPORT_HEADER = [
    "Task Type", "Status", "Documents", "Error", "Started At", "Completed At", "Created At",
]

KNOWLEDGE_TOPICS_REPORT_HEADER = [
    "Topic", "Category", "Mentions", "Connected Documents", "Contributors",
]

USAGE_REPORT_HEADER = [
    "Question", "Answer", "Search Method", "Confidence (%)", "Response Time (ms)", "Asked At",
]

# Used instead of USAGE_REPORT_HEADER when a platform Admin exports a
# company's WHOLE usage report (whole_workspace=True below) - see
# get_usage_report_rows()'s docstring for why content is never
# included in that cross-member case.
USAGE_REPORT_HEADER_WORKSPACE = [
    "Owner", "Search Method", "Confidence (%)", "Response Time (ms)", "Asked At",
]

COMPARISON_REPORT_HEADER = [
    "Metric", "Current Period", "Previous Period", "Change (%)",
]

AI_TASK_RESULTS_HEADER = [
    "Rank", "Document", "Score", "Title", "Summary", "Citations",
]


def get_documents_report_rows(user, organization=None, scope_to_own=False):
    """
    One row per document in the active workspace, most recently
    uploaded first. `organization=None` (Personal Workspace) scopes to
    `user`'s own documents; a real Organization scopes to every
    document in that tenant - same "tenant-shared, not per-uploader"
    rule documents_views.documents_list_view already applies for an
    active organization workspace.

    `scope_to_own=True` further narrows the Organization branch down to
    just `user`'s own documents - for a Member granted Reports access
    via their own Plan-bounded feature toggle rather than the Owner
    rank, who should see only their own activity, never the rest of
    the company's.
    """

    documents = (
        Document.objects.filter(organization=organization, **({"user": user} if scope_to_own else {}))
        if organization is not None
        else Document.objects.filter(user=user, organization__isnull=True)
    ).order_by("-uploaded_at")

    return [
        [
            document.title,
            document.file_type,
            document.file_size,
            document.chunk_count,
            document.uploaded_at.strftime("%Y-%m-%d %H:%M:%S"),
        ]
        for document in documents
    ]


def get_usage_report_rows(user, organization=None, whole_workspace=False):
    """
    One row per question asked in the active workspace, most recent
    first. Personal Workspace (organization=None, whole_workspace has
    no effect there) always filters to `user`'s own rows - QueryLog is
    treated as inherently personal there ("questions I asked" - see
    knowledge_service.get_citation_explorer()'s docstring for the same
    rule) - content (question/answer) included since it's the caller's
    own.

    `whole_workspace=True` (reports_views.py passes this whenever
    `organization is not None` - i.e. every viewer inside a company,
    Owner or a platform Admin auditing it via the admin-only company
    override) scopes to every member's rows in `organization` instead
    of just `user`'s - matching Documents/AI Task Runs' existing
    whole-team behavior for a company workspace, and (for the admin-
    override case specifically) avoiding an Admin picking "Company X"
    from just getting their own almost-always-empty rows, since an
    Admin is rarely a real member. Content is NEVER included in this
    cross-member case, only metadata (owner/method/confidence/response
    time/timestamp), the same "your own content is fine, someone
    else's requires explicit content-viewing - which this app no
    longer offers at all" boundary admin_queries_views.py enforces for
    Admin > Queries.
    """

    if whole_workspace and organization is not None:
        logs = QueryLog.objects.filter(organization=organization).select_related("user").order_by("-created_at")
        return [
            [
                log.user.username,
                log.search_method,
                log.confidence,
                log.response_time_ms,
                log.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            ]
            for log in logs
        ]

    logs = QueryLog.objects.filter(user=user, organization=organization).order_by("-created_at")

    return [
        [
            log.question,
            log.answer,
            log.search_method,
            log.confidence,
            log.response_time_ms,
            log.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        ]
        for log in logs
    ]


QUERIES_REPORT_HEADER_METADATA = [
    "Owner", "Status", "Search Method", "Confidence (%)", "Response Time (ms)", "Sources", "Flagged", "Asked At",
]

QUERIES_REPORT_HEADER_WITH_CONTENT = [
    "Owner", "Question", "Answer", "Status", "Search Method", "Confidence (%)", "Response Time (ms)", "Sources", "Flagged", "Asked At",
]


def get_queries_report_rows(logs, include_content):
    """
    One row per QueryLog in `logs` (already filtered/sorted by the
    caller - see queries_service.filter_and_sort_queries), for the
    Admin > Queries CSV export. `include_content` gates whether
    Question/Answer columns are present at all - the caller is
    responsible for only passing True when the actor holds
    "queries.view_content" (see RAG.views.export_queries_report),
    same privacy boundary as the on-page table.
    """

    from .prompt_templates import is_not_found_answer

    rows = []

    for log in logs:
        status = "No Answer Found" if is_not_found_answer(log.answer) else "Answered"
        common = [
            log.user.username,
        ]
        if include_content:
            common += [log.question, log.answer]
        common += [
            status,
            log.search_method,
            log.confidence,
            log.response_time_ms,
            len(log.sources or []),
            "Yes" if log.is_flagged else "No",
            log.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        ]
        rows.append(common)

    return rows


def get_comparison_report_rows(comparison_data):
    """
    One row per metric in `comparison_data["rows"]` (see
    stats_service.get_comparison_report_data) - the CSV counterpart to
    the on-page Comparative Report table on the Reports page.
    """

    return [
        [row["label"], row["current"], row["previous"], row["change_pct"]]
        for row in comparison_data["rows"]
    ]


def get_ai_task_result_rows(run):
    """
    One row per AITaskResult belonging to `run`, in its own display
    order (see AITaskResult.Meta.ordering). Generic across all 8 task
    types - reads only the common fields (rank/document/score/title/
    summary/citations), never task-specific `data` keys, since the
    on-page results table is where the rich, per-type rendering lives;
    this CSV is the flat, universal export.
    """

    rows = []

    for result in run.results.select_related("document").all():
        document_label = result.document.title if result.document_id else (result.title or "(overall)")
        citations_label = ", ".join(f"[{c['number']}] {c['document']}" for c in (result.citations or []))

        rows.append([
            result.rank if result.rank is not None else "",
            document_label,
            result.score if result.score is not None else "",
            result.title,
            result.summary,
            citations_label,
        ])

    return rows


def get_ai_task_runs_report_rows(user, organization=None, scope_to_own=False):
    """
    One row per AITaskRun in the active workspace, most recent first -
    the AI Tasks counterpart to the Usage Report, but at the run level
    rather than the per-result level get_ai_task_result_rows()/
    ai_task_export already covers for a single run. Same tenant-shared
    scoping as ai_tasks_views.ai_task_history_view for an active
    organization workspace.

    `scope_to_own` mirrors get_documents_report_rows()'s param - a
    Member granted Reports access via their own Plan-bounded feature
    toggle sees only their own runs, never the rest of the company's.
    """

    runs = (
        AITaskRun.objects.filter(organization=organization, **({"user": user} if scope_to_own else {}))
        if organization is not None
        else AITaskRun.objects.filter(user=user, organization__isnull=True)
    ).order_by("-created_at")

    return [
        [
            run.get_task_type_display(),
            run.get_status_display(),
            run.document_count,
            run.error_message,
            run.started_at.strftime("%Y-%m-%d %H:%M:%S") if run.started_at else "",
            run.completed_at.strftime("%Y-%m-%d %H:%M:%S") if run.completed_at else "",
            run.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        ]
        for run in runs
    ]


def get_knowledge_topics_report_rows(user, organization=None):
    """
    One row per Topic visible to `user` in the active workspace (see
    knowledge_service.list_all_topics - the same accessible-scoped,
    cross-uploader-merged Topic list Explore Topics itself uses), most
    mentioned first.
    """

    from .knowledge_service import list_all_topics

    return [
        [
            topic["display_name"],
            topic["entity_type"],
            topic["mention_count"],
            topic["document_count"],
            topic["member_count"],
        ]
        for topic in list_all_topics(user, organization=organization)
    ]
