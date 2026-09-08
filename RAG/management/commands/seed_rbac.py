"""
Seed default RBAC permissions/roles and backfill role assignments for
existing users. Idempotent - safe to re-run any time (e.g. after
adding a new permission to DEFAULT_PERMISSIONS). Re-running never
resets a built-in role's live permission set: DEFAULT_ROLES'
permission list is only applied the moment a role is first created,
so an Admin's edits made via Admin > Roles (or the Django Admin) to
the built-in "user" role survive future re-runs instead of being
silently reverted to the hardcoded defaults.

Usage, after migrating the RBAC models:

    python manage.py migrate RAG
    python manage.py seed_rbac

Existing accounts are backfilled once, using is_superuser/is_staff
purely as a one-time bootstrap signal (superuser or staff -> Admin,
everyone else -> User). Every check after this point reads
Role/Permission via RAG.services.permission_service, never
is_staff/is_superuser directly.

Admin is the sole built-in role with a hardcoded permission bypass
(Role.has_permission's bypass in RAG/models.py). "Super Admin" (seeded
below) is a *different* thing from the "super_admin" role
RAG/migrations/0016_remove_super_admin_role.py once removed - that was
a second hardcoded bypass tier; this is an ordinary dynamic role
(no bypass) holding every permission except
permission_service.SENSITIVE_PERMISSIONS, seeded here the same way any
other role is. See RAG/management/commands/seed_super_admin.py for
seeding the one built-in account that holds it.
"""

from django.contrib.auth.models import User
from django.core.management.base import BaseCommand
from django.db import transaction

from RAG.models import Permission, Role, UserRole
from RAG.services.permission_service import ADMIN, SENSITIVE_PERMISSIONS, SUPER_ADMIN, USER

# (codename, name, description) - namespaced "<area>.<action>" so new
# areas can be added without colliding with existing codenames.
DEFAULT_PERMISSIONS = [
    ("users.view_all", "View all users", "See the full user list and their assigned roles."),
    ("users.delete", "Delete users", "Permanently remove a user account."),
    ("users.suspend", "Suspend users", "Deactivate/reactivate a user account."),
    ("users.assign_role", "Assign roles", "Change a user's assigned role."),

    ("roles.manage", "Manage roles", "Create, edit, and delete custom roles, and assign their permissions."),

    # Platform-wide oversight of every organization/tenant - this is
    # the "Super Admin" capability the multi-tenancy spec asks for,
    # expressed as ordinary permissions on the existing Admin role
    # rather than a second top-tier role. See
    # RAG/services/org_permission_service.py's module docstring for
    # the full rationale. Org-scoped equivalents (a member managing
    # their OWN organization) go through org_permission_service
    # instead - these codenames only ever gate the platform-wide,
    # cross-organization view.
    ("organizations.view_all", "View all organizations", "See every organization on the platform, regardless of membership."),
    ("organizations.manage", "Manage organizations", "Create, suspend, and delete any organization, and view its full detail/statistics."),

    ("billing.manage_plans", "Manage billing plans", "Create/edit/deactivate Plans and assign a Plan to any organization."),
    ("billing.view_all", "View all organization billing", "View usage and plan assignment across every organization, for support/oversight."),

    ("documents.view_all", "View all documents", "See document metadata (title/owner/size/status) across every user."),
    ("documents.delete_any", "Delete any document", "Delete a document owned by any user, not just your own."),
    ("documents.share", "Share documents", "Share your own documents with a specific user or role."),
    ("documents.manage_org_library", "Manage organization library", "Add or remove documents from the org-wide Organization Library."),

    ("analytics.view_all", "View all analytics", "View workspace-wide analytics, not just your own usage."),

    ("system.view_health", "View system health", "View database/pgvector/LLM/embedding health checks."),
    ("system.view_ai_logs", "View AI logs", "View the AI Logs execution-trace explorer - per-request/run stage timing, provider/token usage, and errors for both Ask AI and AI Tasks."),
    ("system.view_storage", "View storage", "View total storage used across the workspace."),
    ("system.view_embeddings", "View embeddings", "View embedding coverage/status."),
    ("system.view_api_status", "View API status", "View LLM/API provider configuration status."),

    ("queries.view_all_logs", "View all query logs", "View every user's Ask AI query log metadata (owner, confidence, response time, timestamp) - not the question/answer content."),
    ("queries.view_content", "View query content", "View the actual question and answer text of other users' queries. A further-gated tier on top of queries.view_all_logs, intended for explicitly authorized auditing only."),
    ("activity.view_all_logs", "View all activity logs", "View a workspace-wide activity log - every page visit and business event (uploads, deletions, role changes, logins, ...), without IP address or geolocation."),
    ("activity.view_ip_location", "View IP & location data", "View the IP address and geolocation (city/region/country) captured for each activity log event. A further-gated tier on top of activity.view_all_logs, for explicitly authorized auditing only."),

    ("settings.manage_llm", "Manage LLM configuration", "View/change the active LLM provider, model, and answer temperature."),
    ("settings.manage_embedding", "Manage embedding model", "View the embedding model configuration (read-only - changing it needs re-embedding + a migration, not just a config flip)."),
    ("settings.manage_chunking", "Manage chunk configuration", "View/change chunk size/overlap and retrieval top-K."),
    ("settings.manage_retrieval", "Manage advanced retrieval", "View/change Query Expansion, HyDE, Multi-Query, Dynamic Top-K, Reranking, and Context Compression."),
    ("settings.manage_api_keys", "Manage API keys", "View/rotate provider API keys."),
    ("settings.manage_database", "Manage database configuration", "View database connection configuration (read-only - editing it live would mean writing the new value through the connection being replaced)."),

    ("notifications.send_announcement", "Send system announcements", "Send a workspace-wide notification/announcement to every user."),
    ("notifications.view_all", "View all notifications", "View notification delivery history across every user, for support/troubleshooting."),

    # Page-level access - these gate whether a role can open a page at
    # all (and whether its nav item even renders), distinct from the
    # cross-user data-scope permissions above (e.g. "documents.view_all"
    # governs seeing *everyone's* documents; "pages.documents" governs
    # whether this role can use the Documents page for its own
    # documents in the first place). Overview (Admin or User) is
    # deliberately NOT in this list - it's never permission-gated as a
    # whole page, the same way Profile isn't; see
    # permission_service.get_dashboard_url_for_user.
    ("pages.documents", "Access Documents", "Upload, view, and manage your own documents."),
    ("pages.knowledge_base", "Access Knowledge Base", "Browse entities, relationships, the knowledge graph, and citations."),
    ("pages.ask_ai", "Access AI Search", "Ask questions and view your own search history."),
    ("pages.analytics", "Access Analytics", "View your own usage analytics."),
    ("pages.reports", "Access Reports", "Export document and usage reports."),
    ("pages.ai_tasks", "Access AI Tasks", "Run guided AI operations (analyze, compare, summarize, extract, validate, find similar, organize, generate reports) over selected documents."),
]

# Baseline page-access permissions every regular "user" role account
# gets out of the box - matches what every logged-in account could
# already reach before navigation became permission-gated, so seeding
# this doesn't regress existing behavior. Admin doesn't need an entry
# here - it gets "__all__" below.
USER_DEFAULT_PERMISSIONS = [
    "pages.documents",
    "pages.knowledge_base",
    "pages.ask_ai",
    "pages.analytics",
    "pages.reports",
    "documents.share",
    "pages.ai_tasks",
]

# Every permission except the ones flagged SENSITIVE_PERMISSIONS (raw
# query content, precise IP/geolocation) - full platform-wide control
# (every organization, every user, every system/settings surface) with
# no path to actual private content. Unlike Admin, this role has no
# Role.has_permission() bypass, so this list IS its access - and,
# unlike Admin's "__all__", it's fixed at the moment this role is first
# created (see the `if created:` guard below): a permission added to
# DEFAULT_PERMISSIONS later won't automatically reach an
# already-created Super Admin role, same one-time-attach behavior any
# other non-Admin role already has.
SUPER_ADMIN_PERMISSIONS = [c for c, _, _ in DEFAULT_PERMISSIONS if c not in SENSITIVE_PERMISSIONS]

# role slug -> (display name, permission codenames or "__all__")
DEFAULT_ROLES = {
    ADMIN: {
        "name": "Admin",
        "permissions": "__all__",
    },
    SUPER_ADMIN: {
        "name": "Super Admin",
        "permissions": SUPER_ADMIN_PERMISSIONS,
    },
    USER: {
        "name": "User",
        "permissions": USER_DEFAULT_PERMISSIONS,
    },
}


class Command(BaseCommand):
    help = "Seed default RBAC permissions/roles and backfill role assignments for existing users."

    @transaction.atomic
    def handle(self, *args, **options):

        permissions_by_codename = {}

        for codename, name, description in DEFAULT_PERMISSIONS:
            permission, created = Permission.objects.get_or_create(
                codename=codename,
                defaults={"name": name, "description": description},
            )
            permissions_by_codename[codename] = permission
            self.stdout.write(f"{'Created' if created else 'Exists'} permission: {codename}")

        roles_by_slug = {}

        for slug, config in DEFAULT_ROLES.items():
            role, created = Role.objects.get_or_create(
                slug=slug,
                defaults={"name": config["name"], "is_system": True},
            )
            roles_by_slug[slug] = role

            if created:
                if config["permissions"] == "__all__":
                    role.permissions.set(permissions_by_codename.values())
                else:
                    role.permissions.set([permissions_by_codename[code] for code in config["permissions"]])

            self.stdout.write(f"{'Created' if created else 'Exists'} role: {slug}")

        default_role = roles_by_slug[USER]
        assigned = 0

        for user in User.objects.all():

            if hasattr(user, "role_assignment"):
                continue

            if user.is_superuser or user.is_staff:
                role = roles_by_slug[ADMIN]
            else:
                role = default_role

            UserRole.objects.create(user=user, role=role)
            assigned += 1

        self.stdout.write(self.style.SUCCESS(
            f"RBAC seed complete. Backfilled role assignments for {assigned} user(s)."
        ))
