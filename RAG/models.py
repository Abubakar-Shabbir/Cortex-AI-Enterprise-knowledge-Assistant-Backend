from django.contrib.auth.models import User
from django.db import models
from pgvector.django import HnswIndex, VectorField
from django.conf import settings

class Document(models.Model):

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="documents"
    )

    title = models.CharField(
        max_length=200
    )

    file = models.FileField(
        upload_to="documents/"
    )

    uploaded_at = models.DateTimeField(
        auto_now_add=True
    )

    file_hash = models.CharField(
        max_length=64,
        blank=True,
        default="",
        db_index=True
    )
    file_size = models.BigIntegerField(
        default=0
    )

    file_type = models.CharField(
        max_length=20,
        blank=True
    )

    chunk_count = models.PositiveIntegerField(
        default=0
    )

    class ProcessingStatus(models.TextChoices):
        PENDING = "pending", "Pending"
        PROCESSING = "processing", "Processing"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"

    processing_status = models.CharField(
        max_length=20,
        choices=ProcessingStatus.choices,
        default=ProcessingStatus.PENDING,
        help_text="Set by upload_service.process_uploaded_document() - "
                   "PENDING until the Embed button is clicked (documents_view.document_embed).",
    )

    description = models.TextField(blank=True, default="")

    is_org_library = models.BooleanField(
        default=False,
        db_index=True,
        help_text="Admin-managed Organization Library membership - visible/retrievable "
                   "to every user, not just the owner. See document_access_service. Not "
                   "related to the `organization` field below despite the name overlap - "
                   "this predates multi-tenancy and means 'shared workspace-wide', while "
                   "`organization` means 'belongs to this one tenant'. Kept as-is rather "
                   "than renamed for now; see the Organization model's docstring.",
    )

    organization = models.ForeignKey(
        "Organization",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="documents",
        db_index=True,
        help_text="NULL means this document lives in its owner's Personal Workspace - "
                   "today's exact pre-multi-tenancy behavior, unchanged. Set only when "
                   "created while an organization workspace was active; every org-scoped "
                   "query filters by this field, never by `user` alone, for a document "
                   "that has it set. CASCADE: deleting an organization deletes its "
                   "documents, matching how deleting a Document's owning User already "
                   "cascades to it - a tenant's data doesn't outlive the tenant.",
    )

    is_archived = models.BooleanField(default=False, db_index=True)
    archived_at = models.DateTimeField(null=True, blank=True)

    category = models.ForeignKey(
        "Category",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="documents",
    )

    tags = models.ManyToManyField("Tag", blank=True, related_name="documents")

    version_number = models.PositiveIntegerField(
        default=1,
        help_text="Current version. Prior versions are snapshotted into DocumentVersion "
                   "by upload_service.upload_new_version() before this is incremented.",
    )

    processing_attempts = models.IntegerField(default=0)
    processing_error = models.TextField(blank=True, default="")
    processing_started_at = models.DateTimeField(null=True, blank=True)
    processing_completed_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return self.title


class Category(models.Model):
    """A single-valued grouping, scoped per user - each user keeps their own category vocabulary, same as Tag, so categorizing a shared/org document never leaks a category name to its other viewers."""

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="categories")
    name = models.CharField(max_length=80)
    slug = models.SlugField(max_length=90)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("user", "slug")
        ordering = ["name"]

    def __str__(self):
        return self.name


class Tag(models.Model):
    """A multi-valued label, scoped per user - see Category's docstring for why."""

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="tags")
    name = models.CharField(max_length=50)
    slug = models.SlugField(max_length=60)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("user", "slug")
        ordering = ["name"]

    def __str__(self):
        return self.name


class Collection(models.Model):
    """A personal folder. The owner need not own every document inside it - only be able to access it (checked in collections_service, not a DB constraint), since filing an org/shared document into your own folder is a legitimate use case."""

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="collections")
    name = models.CharField(max_length=150)
    description = models.CharField(max_length=255, blank=True, default="")
    documents = models.ManyToManyField(
        Document, through="CollectionDocument", related_name="collections"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("user", "name")
        ordering = ["name"]

    def __str__(self):
        return self.name


class CollectionDocument(models.Model):
    collection = models.ForeignKey(Collection, on_delete=models.CASCADE, related_name="items")
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="collection_memberships")
    added_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("collection", "document")


class Favorite(models.Model):
    """Any accessible document can be favorited, not just an owned one - see document_access_service.can_view_document."""

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="favorite_documents")
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="favorited_by")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("user", "document")


class DocumentVersion(models.Model):
    """
    History-only snapshot of a document's PREVIOUS active file, written
    right before that file gets replaced (upload_service.upload_new_version).
    Document.file/.file_hash/.file_size/.file_type/.version_number always
    describe the CURRENT version; this table only ever grows - there is no
    "restore as current" action.
    """

    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="versions")
    version_number = models.PositiveIntegerField()
    file = models.FileField(upload_to="documents/versions/")
    file_hash = models.CharField(max_length=64, blank=True, default="")
    file_size = models.BigIntegerField(default=0)
    file_type = models.CharField(max_length=20, blank=True)
    replaced_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("document", "version_number")
        ordering = ["-version_number"]


class DocumentShare(models.Model):
    """
    Exactly one of shared_with_user / shared_with_role / invited_email is
    set (enforced by the CheckConstraint below). Grants view + download
    only - never delete/re-version/manage-shares; those stay owner-only
    regardless of any share (see document_access_service.can_edit_document).

    invited_email is the pending state for "shared with someone who
    doesn't have an account yet" - it grants nothing by itself
    (document_access_service never looks at it), and is the ONLY field
    here that ever changes after creation: RAG.services.otp_service.
    verify_otp() converts it to a real shared_with_user the moment that
    exact email address is verified via OTP, which is also the moment
    ownership of the address is actually proven - see that function's
    own docstring for the full rationale.
    """

    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="shares")
    shared_with_user = models.ForeignKey(
        User, on_delete=models.CASCADE, null=True, blank=True, related_name="document_shares_received"
    )
    shared_with_role = models.ForeignKey(
        "Role", on_delete=models.CASCADE, null=True, blank=True, related_name="document_shares"
    )
    invited_email = models.CharField(
        max_length=254, blank=True, default="",
        help_text="Set only while pending - an email with no matching account yet. Cleared (and shared_with_user set) once that address completes signup + OTP verification.",
    )
    shared_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name="+")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.CheckConstraint(
                condition=(
                    models.Q(shared_with_user__isnull=False, shared_with_role__isnull=True, invited_email="")
                    | models.Q(shared_with_user__isnull=True, shared_with_role__isnull=False, invited_email="")
                    | models.Q(shared_with_user__isnull=True, shared_with_role__isnull=True, invited_email__gt="")
                ),
                name="documentshare_exactly_one_target",
            ),
            # A plain unique_together on ("document", "invited_email")
            # would treat every non-invite row's blank "" the same as
            # any other row's blank "" - Postgres uniqueness doesn't
            # exempt empty string the way it exempts NULL, so two
            # ordinary user/role shares on the same document would
            # collide with each other. Scoped (partial) instead:
            # uniqueness only applies among rows that are actually a
            # pending invite (invited_email not blank).
            models.UniqueConstraint(
                fields=["document", "invited_email"],
                condition=models.Q(invited_email__gt=""),
                name="documentshare_unique_pending_invite",
            ),
        ]
        unique_together = [
            ("document", "shared_with_user"),
            ("document", "shared_with_role"),
        ]


class DocumentAccessLog(models.Model):
    """Powers Recent Documents - written on document_download/document_preview, not on every list-page render."""

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="document_accesses")
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="access_logs")
    accessed_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [models.Index(fields=["user", "-accessed_at"])]


class DocumentChunk(models.Model):

    document = models.ForeignKey(
        Document,
        on_delete=models.CASCADE,
        related_name="chunks"
    )

    content = models.TextField()

    chunk_number = models.PositiveIntegerField()

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    class Meta:
        ordering = ["chunk_number"]

    def __str__(self):
        return f"{self.document.title} - Chunk {self.chunk_number}"


class ChunkEmbedding(models.Model):

    chunk = models.OneToOneField(
        DocumentChunk,
        on_delete=models.CASCADE,
        related_name="vector"
    )

    embedding = VectorField(
        dimensions=settings.EMBEDDING_DIMENSION
    )

    embedding_model = models.CharField(
        max_length=100,
        default="all-MiniLM-L6-v2"
    )

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    class Meta:
        # HNSW, not IVFFlat: IVFFlat needs representative data present
        # at index-creation time to pick good cluster centroids and a
        # tuned `lists` parameter, and degrades badly on a small/empty
        # table; HNSW needs neither and searches correctly from the
        # first row. opclasses matches L2Distance - the only distance
        # function every retrieval query in this project actually
        # uses (retrieval_service._vector_similarity_search) - an
        # index built for a different operator wouldn't be used by
        # these queries at all.
        indexes = [
            HnswIndex(
                name="chunkembedding_embedding_hnsw",
                fields=["embedding"],
                m=16,
                ef_construction=64,
                opclasses=["vector_l2_ops"],
            )
        ]

    def __str__(self):
        return (
            f"{self.chunk.document.title} "
            f"- Embedding ({self.embedding_model})"
        )


class QueryLog(models.Model):
    """
    Records every question asked so the
    Search History and Analytics pages can
    show real usage instead of placeholders.
    """

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="query_logs"
    )

    organization = models.ForeignKey(
        "Organization",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="query_logs",
        db_index=True,
        help_text="NULL means this question was asked in the user's Personal Workspace - "
                   "see Document.organization's help_text for the shared convention every "
                   "organization-aware model in this file follows.",
    )

    question = models.TextField()

    answer = models.TextField()

    sources = models.JSONField(
        default=list,
        blank=True
    )

    structured_data = models.JSONField(
        default=dict,
        blank=True,
        help_text="Non-plain-text answer extras at the time this question was asked - "
                   "{'key_points': [...], 'table': {...} or None}. Persisted so replaying "
                   "a question from Search History (RAG.views.ask_ai's ?log_id= path) can "
                   "show the same structured findings the live answer did, instead of just "
                   "the answer text. A streamed answer (RAG.views.ask_ai_stream) always "
                   "writes {'key_points': [], 'table': None} here since it's plain text.",
    )

    search_method = models.CharField(
        max_length=50,
        default="Hybrid (Vector + BM25)"
    )

    response_time_ms = models.PositiveIntegerField(
        default=0
    )

    confidence = models.PositiveSmallIntegerField(
        default=0
    )

    created_at = models.DateTimeField(
        auto_now_add=True,
        db_index=True
    )

    is_flagged = models.BooleanField(
        default=False,
        help_text="Pinned for follow-up in Admin > Queries. A shared review flag, not tied to any one admin.",
    )

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["organization", "created_at"])]

    def __str__(self):
        return f"{self.user.username}: {self.question[:50]}"


class Entity(models.Model):
    """
    A normalized, deduplicated entity mentioned in one or more of
    the user's document chunks. Entity types are a free-form string
    rather than a fixed choices list so new types (from the LLM
    extractor or future extractors) don't require a migration.
    """

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="entities"
    )

    organization = models.ForeignKey(
        "Organization",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="entities",
        db_index=True,
        help_text="NULL means this entity was extracted from a Personal Workspace document - see Document.organization's help_text for the shared convention.",
    )

    name = models.CharField(
        max_length=255,
        db_index=True
    )

    display_name = models.CharField(
        max_length=255
    )

    entity_type = models.CharField(
        max_length=50,
        db_index=True,
        default="MISC"
    )

    mention_count = models.PositiveIntegerField(
        default=0
    )

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    class Meta:
        ordering = ["name"]
        unique_together = ("user", "organization", "name", "entity_type")

    def __str__(self):
        return f"{self.display_name} ({self.entity_type})"


class EntityMention(models.Model):
    """
    Links an Entity to the DocumentChunk it was extracted from, so
    graph retrieval can pull the supporting chunk content for a
    matched entity.
    """

    entity = models.ForeignKey(
        Entity,
        on_delete=models.CASCADE,
        related_name="mentions"
    )

    chunk = models.ForeignKey(
        DocumentChunk,
        on_delete=models.CASCADE,
        related_name="entity_mentions"
    )

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    class Meta:
        unique_together = ("entity", "chunk")

    def __str__(self):
        return f"{self.entity.display_name} in {self.chunk}"


class Relationship(models.Model):
    """
    A directed, deduplicated edge between two entities. Re-extracting
    the same (source, relation, target) triple from another chunk
    increments weight instead of creating a duplicate row.
    """

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="relationships"
    )

    organization = models.ForeignKey(
        "Organization",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="relationships",
        db_index=True,
        help_text="NULL means this relationship was extracted from a Personal Workspace document - see Document.organization's help_text for the shared convention.",
    )

    source = models.ForeignKey(
        Entity,
        on_delete=models.CASCADE,
        related_name="outgoing_relationships"
    )

    target = models.ForeignKey(
        Entity,
        on_delete=models.CASCADE,
        related_name="incoming_relationships"
    )

    relation_type = models.CharField(
        max_length=100,
        db_index=True
    )

    context = models.TextField(
        blank=True,
        default=""
    )

    weight = models.PositiveIntegerField(
        default=1
    )

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    updated_at = models.DateTimeField(
        auto_now=True
    )

    class Meta:
        ordering = ["-weight"]
        unique_together = ("user", "source", "target", "relation_type")

    def __str__(self):
        return f"{self.source.display_name} -[{self.relation_type}]-> {self.target.display_name}"


# ============================================================
# RBAC
# ============================================================
#
# Role/Permission/UserRole are the single source of truth for
# authorization from here on - see RAG/services/permission_service.py
# for the read API and RAG/decorators.py / RAG/middleware.py for how
# views enforce it. is_staff/is_superuser are only ever read once, as
# the seed data source for existing accounts in
# RAG/management/commands/seed_rbac.py; no other code should branch on
# them going forward.
#
# Admin is the sole role with a hardcoded bypass (Role.has_permission
# below) - it always has full access regardless of its M2M permission
# rows. "Super Admin" (below) is NOT a second bypass tier - it's an
# ordinary dynamic role like "user", seeded (RAG/management/commands/
# seed_rbac.py) with every permission except the ones in
# permission_service.SENSITIVE_PERMISSIONS (raw query content, precise
# IP/geolocation), so its access is entirely visible/auditable via its
# own M2M row set rather than a hidden bypass. Every role other than
# Admin - "user", "super_admin", and any future custom role - is fully
# dynamic: created, edited, and deleted through Admin > Roles
# (RAG.views.admin_roles_view), with permissions assigned per role,
# not hardcoded per feature.

ADMIN_ROLE_SLUG = "admin"
SUPER_ADMIN_ROLE_SLUG = "super_admin"
USER_ROLE_SLUG = "user"


class Permission(models.Model):
    """
    A single grantable capability, namespaced as "<area>.<action>"
    (e.g. "users.suspend") so new areas can be added without colliding
    with existing codenames.
    """

    codename = models.CharField(
        max_length=100,
        unique=True
    )

    name = models.CharField(
        max_length=150
    )

    description = models.CharField(
        max_length=255,
        blank=True
    )

    class Meta:
        ordering = ["codename"]

    def __str__(self):
        return self.codename

    @property
    def namespace(self):
        """The "<area>" half of "<area>.<action>" - lets the Role Management UI group permissions by area without a second lookup table."""
        return self.codename.split(".")[0]


class Role(models.Model):
    """
    A named bundle of Permissions. Adding a future role (Manager, HR,
    Auditor, ...) means creating a Role row and attaching Permissions
    to it - no code changes required, since every permission check
    goes through Role.has_permission() / permission_service, never a
    hardcoded role name.
    """

    name = models.CharField(
        max_length=100,
        unique=True
    )

    slug = models.SlugField(
        max_length=100,
        unique=True
    )

    description = models.CharField(
        max_length=255,
        blank=True
    )

    permissions = models.ManyToManyField(
        Permission,
        blank=True,
        related_name="roles"
    )

    is_system = models.BooleanField(
        default=False,
        help_text="Built-in role (admin/user) seeded by seed_rbac - not meant to be deleted.",
    )

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return self.name

    def has_permission(self, codename):
        # Admin always has full system access by design (not just at
        # seed time) - a permission added months from now still just
        # works for Admin without anyone remembering to attach it.
        # The M2M table is still kept in sync for Admin (see the
        # Permission post_save signal in RAG/apps.py and
        # RAG.admin.RoleAdmin.save_related) purely so the Role
        # Management UI's checkboxes show the truth, not because this
        # check depends on it.
        if self.slug == ADMIN_ROLE_SLUG:
            return True
        return self.permissions.filter(codename=codename).exists()


class UserRole(models.Model):
    """
    A user's single active role. One-to-one rather than many-to-many:
    this product's roles (Super Admin / Admin / User, and whatever is
    added later) are mutually exclusive tiers, not stackable grants -
    a user needing broader access gets reassigned, not given a second
    role.
    """

    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name="role_assignment"
    )

    role = models.ForeignKey(
        Role,
        on_delete=models.PROTECT,
        related_name="user_assignments"
    )

    assigned_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="+"
    )

    assigned_at = models.DateTimeField(
        auto_now_add=True
    )

    def __str__(self):
        return f"{self.user} -> {self.role}"


class ActivityLog(models.Model):
    """
    A workspace-wide audit trail entry, written by
    RAG.services.activity_log_service.log_activity() - either from a
    specific call site (document deleted, role changed, login, ...) or
    generically for every other request by
    RAG.middleware.RequestActivityMiddleware (action="page.<url_name>"),
    which only fires when nothing more specific already logged that
    same click. Backs the Activity tab of Admin > System Logs
    (RAG.views.admin_system_logs_view). Every row carries the
    request's IP and geolocation (city/region/country/lat/lon) when a
    request was in scope - see RAG.services.geolocation_service.
    """

    actor = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="activity_logs"
    )

    action = models.CharField(
        max_length=50,
        db_index=True,
        help_text='Namespaced event, e.g. "document.deleted", "user.suspended".',
    )

    description = models.CharField(
        max_length=255
    )

    ip_address = models.GenericIPAddressField(
        null=True,
        blank=True,
        help_text="Client IP of the request that triggered this event, via "
                   "activity_log_service.get_client_ip() - None for system-"
                   "initiated events with no request in scope.",
    )

    user_agent = models.TextField(
        blank=True,
        default="",
        help_text="Raw User-Agent header of the triggering request - parsed on "
                   "read by RAG.services.device_intelligence_service.parse_device() "
                   "into device type/browser/OS for login history, never parsed "
                   "at write time so a parser change never needs a backfill.",
    )

    city = models.CharField(max_length=100, blank=True, default="")
    region = models.CharField(max_length=100, blank=True, default="")
    country = models.CharField(max_length=100, blank=True, default="", db_index=True)
    country_code = models.CharField(max_length=8, blank=True, default="")

    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)

    created_at = models.DateTimeField(
        auto_now_add=True,
        db_index=True
    )

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.action}: {self.description}"


class SystemConfiguration(models.Model):
    """
    Singleton row (always pk=1) holding the live-editable subset of RAG
    pipeline configuration - see RAG/services/system_config_service.py,
    which is the only code that should read/write this model directly.
    Everything here is applied on top of settings.py at runtime
    (apply_config_to_settings()), so every existing consumer of
    settings.TOP_K / settings.ENABLE_HYDE / etc. keeps working
    unchanged - this model never replaces settings.py, it overrides it.

    Deliberately NOT included: embedding model (changing it needs a
    migration + re-embedding every existing chunk, not a config flip),
    database connection (editing it live is operationally circular -
    you'd be writing the new value through the connection you're about
    to replace), and raw API keys (secrets belong in environment
    variables, not a database row editable from a browser).
    """

    LLM_PROVIDER_CHOICES = [
        ("openrouter", "OpenRouter"),
        ("groq", "Groq"),
        ("gemini", "Gemini"),
    ]

    llm_provider = models.CharField(max_length=20, choices=LLM_PROVIDER_CHOICES, default="openrouter")

    # Off by default: the configured provider above must be the one
    # that actually answers, or the request fails outright - see
    # settings.LLM_FALLBACK_ENABLED and llm_client._build_chain() for
    # why this can never be a silent, invisible switch to another
    # provider unless an admin explicitly opts in here.
    enable_fallback = models.BooleanField(default=False)

    # Per-provider model selection - kept independent per provider
    # (rather than one shared "model" field) since a model name from
    # one provider is meaningless to another. See
    # RAG.services.llm_client.PROVIDER_REGISTRY for the curated
    # free-model choices the Settings page offers per provider; these
    # fields accept any string, so a value set directly via .eee is
    # never clobbered by the dropdown's curated list.
    openrouter_model = models.CharField(max_length=150, default="nvidia/nemotron-3-super-120b-a12b:free")
    groq_model = models.CharField(max_length=150, default="llama-3.1-8b-instant")
    gemini_model = models.CharField(max_length=150, default="gemini-2.0-flash")

    top_k = models.PositiveSmallIntegerField(default=3)
    answer_temperature = models.FloatField(default=0.2)

    chunk_size = models.PositiveIntegerField(
        default=800,
        help_text="Applies to newly-uploaded documents only - existing chunks aren't retroactively resized.",
    )
    chunk_overlap = models.PositiveIntegerField(default=150)

    enable_query_expansion = models.BooleanField(default=False)
    enable_hyde = models.BooleanField(default=False)
    enable_multi_query = models.BooleanField(default=False)
    multi_query_variants = models.PositiveSmallIntegerField(default=3)

    enable_dynamic_top_k = models.BooleanField(default=True)
    dynamic_top_k_max = models.PositiveSmallIntegerField(default=10)

    enable_reranker = models.BooleanField(default=False)
    reranker_candidate_multiplier = models.PositiveSmallIntegerField(default=3)

    enable_context_compression = models.BooleanField(default=False)
    context_compression_threshold = models.FloatField(default=0.92)

    updated_at = models.DateTimeField(auto_now=True)

    updated_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="+"
    )

    class Meta:
        verbose_name = "System Configuration"

    def __str__(self):
        return "System Configuration"


class AITaskRun(models.Model):
    """
    One guided AI Task run (Select Task -> Select Documents -> Configure
    -> Run -> Review -> Export). Always processed on the in-process
    background thread pool (RAG.tasks.run_ai_task, dispatched via
    RAG.services.task_runner) - unlike document processing, there is no
    inline/synchronous path, since a run can span up to
    settings.AI_TASKS_MAX_DOCUMENTS documents.
    """

    class TaskType(models.TextChoices):
        ANALYZE = "analyze", "Analyze Documents"
        COMPARE = "compare", "Compare Documents"
        SUMMARIZE = "summarize", "Summarize Documents"
        EXTRACT = "extract", "Extract Information"
        VALIDATE = "validate", "Validate Against Reference Documents"
        FIND_SIMILAR = "find_similar", "Find Similar Documents"
        ORGANIZE = "organize", "Organize Documents"
        REPORT = "report", "Generate Reports"

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        RUNNING = "running", "Running"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"
        CANCELLED = "cancelled", "Cancelled"

    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="ai_task_runs"
    )

    organization = models.ForeignKey(
        "Organization",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="ai_task_runs",
        db_index=True,
        help_text="NULL means this run was started in the user's Personal Workspace - see Document.organization's help_text for the shared convention.",
    )

    task_type = models.CharField(max_length=20, choices=TaskType.choices)

    status = models.CharField(
        max_length=20, choices=Status.choices, default=Status.PENDING
    )

    config = models.JSONField(
        default=dict,
        blank=True,
        help_text="Task-specific options from the Configure step (criteria, "
                   "fields, similarity_threshold, ...) - shape depends on "
                   "task_type, validated by ai_tasks_engine_service, not the DB.",
    )

    document_count = models.PositiveIntegerField(
        default=0,
        help_text="Snapshot of the target-document count at creation time, "
                   "after the AI_TASKS_MAX_DOCUMENTS cap was enforced.",
    )

    error_message = models.TextField(blank=True, default="")

    cancel_requested = models.BooleanField(
        default=False,
        help_text="Set by ai_task_cancel() when a user stops a pending/running run. "
                   "execute_run() checks this in _call_llm_json() (the one choke point "
                   "every task handler's LLM calls go through) between calls and unwinds "
                   "to status=CANCELLED - so a stop takes effect within one LLM call's "
                   "worth of latency, not instantly, and never mid-write of a result row.",
    )

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["organization", "created_at"])]

    def __str__(self):
        return f"{self.get_task_type_display()} ({self.id}) - {self.user.username}"


class AITaskRunDocument(models.Model):
    """
    Through model for AITaskRun's document set. TARGET documents are
    the ones being analyzed/compared/summarized/...; REFERENCE
    documents (a job description, a policy, a standard) are only used
    by Validate Against Reference Documents, and optionally Analyze
    Documents, to check TARGET documents against.
    """

    class Role(models.TextChoices):
        TARGET = "target", "Target"
        REFERENCE = "reference", "Reference"

    run = models.ForeignKey(
        AITaskRun, on_delete=models.CASCADE, related_name="run_documents"
    )

    document = models.ForeignKey(
        Document, on_delete=models.CASCADE, related_name="ai_task_run_memberships"
    )

    role = models.CharField(
        max_length=20, choices=Role.choices, default=Role.TARGET
    )

    class Meta:
        unique_together = ("run", "document")
        indexes = [models.Index(fields=["run", "role"])]


class AITaskResult(models.Model):
    """
    One row per result item a run produces. `document` is NULL for a
    corpus-level result (Compare's overall diff narrative, Organize's
    per-group scheme, Generate Reports' final report body, Summarize's
    executive summary, Find Similar's per-cluster label) rather than a
    finding about one specific target document.

    SET_NULL (not CASCADE) on `document` so deleting a document later
    doesn't erase review history the user already exported - the row
    just loses its live document link and falls back to the `title`
    snapshot.
    """

    run = models.ForeignKey(AITaskRun, on_delete=models.CASCADE, related_name="results")

    document = models.ForeignKey(
        Document,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="ai_task_results",
    )

    rank = models.PositiveIntegerField(
        null=True, blank=True,
        help_text="Display order within its group - Analyze ranking, Organize "
                   "group index, Find Similar cluster index. NULL where no "
                   "ordering applies.",
    )

    score = models.FloatField(
        null=True, blank=True,
        help_text="Task-specific numeric score on a consistent 0-100 scale - "
                   "Analyze relevance, Validate compliance percent, Find "
                   "Similar/Organize similarity (cosine similarity * 100). "
                   "NULL where not applicable.",
    )

    title = models.CharField(
        max_length=255, blank=True, default="",
        help_text="Document title snapshot for a per-document row, or a "
                   "heading (e.g. 'Executive Summary') for a corpus-level row.",
    )

    summary = models.TextField(
        blank=True, default="",
        help_text="LLM prose finding for this result, with inline [n] "
                   "citation markers per the citation_service.py convention.",
    )

    data = models.JSONField(
        default=dict, blank=True,
        help_text="Task-specific structured payload (extracted fields, "
                   "violations, combined table, report sections, ...). "
                   "Rendered generically by templates/ai_tasks/_result_row.html "
                   "based on which keys are present.",
    )

    citations = models.JSONField(
        default=list, blank=True,
        help_text="[{number, document, chunk_number, role}, ...] - the subset "
                   "of build_cited_context() source blocks this result's "
                   "summary actually cited.",
    )

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["run_id", "rank", "-score", "id"]
        indexes = [models.Index(fields=["run", "document"])]

    def __str__(self):
        return self.title or f"Result {self.id} for run {self.run_id}"


class AIRequestTrace(models.Model):
    """
    One row per Ask AI question or AI Task run - the shared execution
    trace both features write to (RAG.services.observability_service
    .save_trace() is the one function that creates these; nothing else
    should). Built from the same per-stage timing
    (RAG.services.perf.timed_stage() / RAG.services.trace.record_stage())
    and LLM call metadata (RAG.services.llm_client.get_last_llm_meta())
    both features already produce - this model is where that data
    becomes queryable instead of living only in console log lines.

    Exactly one of `query_log` / `ai_task_run` is set, matching `source`.
    AI Tasks gets one row per RUN, not per per-document LLM call (a run
    makes N+1 LLM calls) - `stages` is a list, so per-item granularity is
    an additive follow-up, not a redesign.
    """

    class Source(models.TextChoices):
        ASK_AI = "ask_ai", "Ask AI"
        AI_TASK = "ai_task", "AI Task"

    class Status(models.TextChoices):
        RUNNING = "running", "Running"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"
        CANCELLED = "cancelled", "Cancelled"

    trace_id = models.CharField(max_length=32, unique=True, db_index=True)

    source = models.CharField(max_length=20, choices=Source.choices, db_index=True)

    user = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, blank=True, related_name="ai_request_traces"
    )

    organization = models.ForeignKey(
        "Organization", on_delete=models.SET_NULL, null=True, blank=True, related_name="ai_request_traces",
        help_text="Denormalized from query_log.organization / ai_task_run.organization at write "
                   "time (observability_service.save_trace()) for cheap per-company/system-wide "
                   "AI usage aggregation - NULL means Personal Workspace, same convention as "
                   "Document.organization.",
    )

    query_log = models.OneToOneField(
        QueryLog, on_delete=models.SET_NULL, null=True, blank=True, related_name="trace"
    )

    ai_task_run = models.OneToOneField(
        AITaskRun, on_delete=models.SET_NULL, null=True, blank=True, related_name="trace"
    )

    status = models.CharField(max_length=20, choices=Status.choices, default=Status.RUNNING, db_index=True)

    provider = models.CharField(max_length=50, blank=True, default="")

    model = models.CharField(max_length=100, blank=True, default="")

    providers_attempted = models.JSONField(
        default=list, blank=True,
        help_text="Ordered list of provider keys actually tried this request/run - "
                   "len() > 1 means a fallback happened.",
    )

    retry_count = models.PositiveIntegerField(default=0)

    prompt_tokens = models.PositiveIntegerField(null=True, blank=True)
    completion_tokens = models.PositiveIntegerField(null=True, blank=True)
    total_tokens = models.PositiveIntegerField(null=True, blank=True)

    llm_latency_ms = models.PositiveIntegerField(null=True, blank=True)

    time_to_first_token_ms = models.PositiveIntegerField(
        null=True, blank=True,
        help_text="Only populated for a streamed Ask AI answer.",
    )

    retrieved_chunks = models.PositiveIntegerField(default=0)

    citation_count = models.PositiveIntegerField(default=0)

    cache_hit = models.BooleanField(null=True, blank=True)

    stages = models.JSONField(
        default=list, blank=True,
        help_text="[{name, duration_ms, ...context}, ...] in execution order - "
                   "the same shape perf.timed_stage() already logs to console.",
    )

    total_duration_ms = models.PositiveIntegerField(default=0)

    bottleneck_stage = models.CharField(max_length=100, blank=True, default="")

    bottleneck_label = models.CharField(max_length=200, blank=True, default="")

    error_type = models.CharField(
        max_length=100, blank=True, default="",
        help_text="A short classified code (e.g. LLMTimeoutError, "
                   "AllProvidersFailedError) - never a raw exception message.",
    )

    error_message = models.TextField(
        blank=True, default="",
        help_text="Sanitized short message for support/debugging - must never "
                   "contain API keys, tokens, or a raw stack trace.",
    )

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["source", "status"]),
            models.Index(fields=["created_at"]),
            models.Index(fields=["organization", "created_at"]),
        ]

    def __str__(self):
        return f"{self.get_source_display()} trace {self.trace_id}"


class ErrorGroup(models.Model):
    """
    Deduplicated errors/warnings captured automatically from the app's
    existing logging calls (RAG.services.error_intelligence_service
    .ErrorCaptureHandler, wired into settings.LOGGING) - every
    logger.warning()/error()/exception() call already made throughout
    RAG/services/*.py, RAG/views.py, RAG/tasks.py (auth, documents,
    retrieval, LLM providers, background jobs, unhandled exceptions)
    flows into this model with zero changes to any of those call sites.

    One row per *distinct* error shape (see `fingerprint`), not one row
    per occurrence - an outage that logs the same warning 500 times
    increments one row's occurrence_count/last_seen/recent_occurrences
    rather than creating 500 rows. This is deliberately read-only
    observability, not a ticketing system - no resolved/acknowledged
    workflow.
    """

    fingerprint = models.CharField(
        max_length=64, unique=True, db_index=True,
        help_text="sha256(logger_name:level:error_type-or-normalized-message) - what groups occurrences together.",
    )

    logger_name = models.CharField(max_length=200, db_index=True, help_text='e.g. "RAG.services.llm_client", "django.request".')

    level = models.CharField(max_length=20, db_index=True, help_text="WARNING / ERROR / CRITICAL - matches the Python logging level name.")

    error_type = models.CharField(max_length=200, blank=True, default="", help_text="Exception class name, if the log call included exc_info - blank for a plain logger.warning() with no exception.")

    message = models.TextField(help_text="Redacted (see error_intelligence_service.redact_secrets()), truncated first-occurrence message - not a full traceback, by design (see the module docstring on what this deliberately doesn't store).")

    occurrence_count = models.PositiveIntegerField(default=1)

    first_seen = models.DateTimeField(auto_now_add=True)

    last_seen = models.DateTimeField(db_index=True)

    recent_occurrences = models.JSONField(
        default=list, blank=True,
        help_text="[{trace_id, timestamp}, ...] - the most recent occurrences only (capped), for correlating back to an AIRequestTrace or a specific request via RAG.services.trace's trace_id / the X-Request-ID response header.",
    )

    class Meta:
        ordering = ["-last_seen"]
        indexes = [
            models.Index(fields=["level", "last_seen"]),
            models.Index(fields=["logger_name", "last_seen"]),
        ]

    def __str__(self):
        return f"[{self.level}] {self.logger_name}: {self.message[:60]}"

    @property
    def severity(self):
        """
        Computed, not stored - a state machine (manually resolved/
        acknowledged errors) is explicitly out of scope; this is just
        "how alarming does this look right now" from level + recent
        volume, recomputed fresh every time it's read.
        """

        from django.utils import timezone

        if self.level == "CRITICAL":
            return "critical"

        recent_and_frequent = self.occurrence_count >= 10 and (timezone.now() - self.last_seen).total_seconds() < 3600

        if self.level == "ERROR" and recent_and_frequent:
            return "critical"

        if self.level == "ERROR":
            return "high"

        if recent_and_frequent:
            return "high"

        return "medium"


class UserProfile(models.Model):
    """
    Extended, enterprise-style profile data on top of auth.User - first/
    last name, username, email, date_joined, and last_login already live
    on User itself and are read directly rather than duplicated here; Role
    comes from the existing RBAC UserRole/Role models (see the "RBAC"
    section above); Account Status is User.is_active.

    Auto-created for every new User by the post_save signal in
    RAG/apps.py (_create_profile_for_new_user); views also call
    get_or_create() defensively for accounts that existed before this
    model did, the same "idempotent, safe to backfill lazily" approach
    RAG/management/commands/seed_rbac.py already uses for role
    assignment.
    """

    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name="profile",
    )

    avatar = models.ImageField(upload_to="avatars/", blank=True, null=True)

    headline = models.CharField(
        max_length=150,
        blank=True,
        default="",
        help_text='One-line professional tagline, e.g. "AI Engineer | RAG Systems | Knowledge Intelligence".',
    )

    phone = models.CharField(max_length=30, blank=True, default="")

    department = models.CharField(max_length=100, blank=True, default="")
    job_title = models.CharField(max_length=150, blank=True, default="")

    employee_id = models.CharField(
        max_length=50,
        blank=True,
        default="",
        help_text="Informational only - not DB-unique, since nothing else in "
                   "this project validates or looks up users by it yet.",
    )

    manager = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="direct_reports",
    )

    team = models.CharField(max_length=100, blank=True, default="")
    location = models.CharField(
        max_length=150,
        blank=True,
        default="",
        help_text="Self-reported office/city location - distinct from the "
                   "IP-derived login location on ActivityLog.",
    )

    TIMEZONE_CHOICES_HELP = "IANA name (e.g. 'America/New_York') - free-form so any valid value saves even if the UI's curated <select> list doesn't cover it."
    timezone = models.CharField(max_length=50, blank=True, default="", help_text=TIMEZONE_CHOICES_HELP)

    # Curated IANA names for the profile form's <select> - not a Django
    # `choices` constraint (the field itself stays free-form, see the
    # help text above), just what the dropdown offers.
    COMMON_TIMEZONES = [
        "UTC",
        "America/New_York", "America/Chicago", "America/Denver", "America/Los_Angeles",
        "America/Sao_Paulo", "America/Mexico_City", "America/Toronto",
        "Europe/London", "Europe/Paris", "Europe/Berlin", "Europe/Madrid", "Europe/Moscow",
        "Africa/Cairo", "Africa/Lagos", "Africa/Johannesburg",
        "Asia/Dubai", "Asia/Karachi", "Asia/Kolkata", "Asia/Dhaka", "Asia/Bangkok",
        "Asia/Shanghai", "Asia/Tokyo", "Asia/Seoul", "Asia/Singapore",
        "Australia/Sydney", "Australia/Perth",
        "Pacific/Auckland",
    ]

    LANGUAGE_CHOICES = [
        ("en", "English"),
        ("es", "Spanish"),
        ("fr", "French"),
        ("de", "German"),
        ("hi", "Hindi"),
        ("zh", "Chinese"),
        ("ja", "Japanese"),
        ("pt", "Portuguese"),
        ("ar", "Arabic"),
    ]
    language = models.CharField(max_length=10, choices=LANGUAGE_CHOICES, blank=True, default="en")

    skills = models.JSONField(default=list, blank=True, help_text="List of plain strings.")
    certifications = models.JSONField(default=list, blank=True, help_text="List of plain strings.")

    linkedin_url = models.URLField(blank=True, default="")
    github_url = models.URLField(blank=True, default="")
    portfolio_url = models.URLField(blank=True, default="")

    class Visibility(models.TextChoices):
        PUBLIC = "public", "Public"
        TEAM = "team", "Team"
        PRIVATE = "private", "Private"

    profile_visibility = models.CharField(
        max_length=10,
        choices=Visibility.choices,
        default=Visibility.PRIVATE,
        help_text="Stored preference only - there is no peer-facing profile "
                   "viewing page yet to enforce it against; Admin views "
                   "always see the full profile regardless (existing RBAC "
                   "bypass pattern, e.g. Role.has_permission).",
    )

    updated_at = models.DateTimeField(auto_now=True)

    email_verified = models.BooleanField(
        default=False,
        help_text="True once the account holder has completed email OTP "
                   "verification at signup (see RAG.services.otp_service). "
                   "Pre-existing accounts were backfilled to True by a "
                   "one-time data migration when this field was added - "
                   "only new signups actually go through OTP.",
    )

    class AccountType(models.TextChoices):
        PERSONAL = "personal", "Personal"
        COMPANY = "company", "Company"

    account_type = models.CharField(
        max_length=10,
        choices=AccountType.choices,
        default=AccountType.PERSONAL,
        help_text="Decided once, during signup (or by accepting an "
                   "organization invitation), never by a workspace "
                   "switcher - see auth_views.signup()/org_invitation_service"
                   ".accept_invitation(). A PERSONAL account never holds an "
                   "OrganizationMembership; a COMPANY account's requests are "
                   "always scoped to one of its organizations, never to "
                   "Personal Workspace - org_permission_service."
                   "resolve_request_organization() enforces both directions "
                   "server-side, so this field (not any frontend state) is "
                   "the actual source of truth an authorization decision "
                   "reads from.",
    )

    ai_credits_balance = models.PositiveIntegerField(
        null=True, blank=True, default=None,
        help_text="Personal-Workspace mirror of Organization.ai_credits_balance - a completely "
                   "separate credit pool, never shared with any Company workspace this same "
                   "user might also belong to. NULL = never topped up / no Personal Plan "
                   "assigned yet (unlimited, same 'absence means unlimited' convention).",
    )

    must_change_password = models.BooleanField(
        default=False,
        help_text="Set when org_member_registration_service.register_company_member() "
                   "creates or reactivates a company-registered member's account with a "
                   "system-generated password (see that module for the full flow) - "
                   "never set for a self-service Personal/Company signup, which chooses "
                   "its own password up front. RAG.api.auth_views.login_view() reports "
                   "this in the session payload; the SPA blocks every route except the "
                   "forced password-change page until it's cleared.",
    )

    def __str__(self):
        return f"{self.user.username}'s profile"


class EmailOTP(models.Model):
    """
    A one-time verification code emailed to a user - currently only used
    for signup verification, but `purpose` is namespaced so a future use
    (e.g. verifying a changed email address) needs no schema change.

    The code itself is never stored in plaintext - `code_hash` uses
    Django's own password hasher (django.contrib.auth.hashers.
    make_password()/check_password()), the same PBKDF2 machinery real
    passwords use, so this module introduces no new cryptographic code.
    The plaintext code exists only as a local variable in
    RAG.services.otp_service.generate_and_send_otp() and as an argument
    to the backgrounded email-send task - never logged, never returned
    in any response.
    """

    class Purpose(models.TextChoices):
        SIGNUP = "signup", "Signup Verification"

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="email_otps")

    purpose = models.CharField(max_length=20, choices=Purpose.choices, default=Purpose.SIGNUP)

    code_hash = models.CharField(max_length=128)

    attempt_count = models.PositiveSmallIntegerField(
        default=0,
        help_text="Incremented on every failed verify attempt against this "
                   "row - a durable, per-row cap (RAG.services.otp_service."
                   "MAX_OTP_ATTEMPTS) on top of the cache-based cross-row "
                   "rate limit in rate_limit_service.",
    )

    is_used = models.BooleanField(default=False)
    used_at = models.DateTimeField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField()

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["user", "purpose", "is_used", "-created_at"])]

    def __str__(self):
        return f"{self.get_purpose_display()} OTP for {self.user.username}"


class Notification(models.Model):
    """
    A single recipient-facing notification - distinct from ActivityLog
    (a system-wide audit trail of what an *actor* did, admin-facing) and
    never populated from it. This is a per-user inbox: "X shared a
    document with you", read/unread, with an optional emailed copy.

    `notification_type` is namespaced "<area>.<event>" (e.g.
    "document.shared", "ai_task.completed", "account.password_changed")
    exactly like Permission.codename and ActivityLog.action, so new
    event types never require a schema change - just a new string and,
    optionally, an icon mapping in RAG.views._NOTIFICATION_ICONS.
    """

    recipient = models.ForeignKey(User, on_delete=models.CASCADE, related_name="notifications")

    actor = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, blank=True, related_name="+",
        help_text="Who caused this notification, if a specific user did (e.g. who shared the document). Blank for system-generated events.",
    )

    notification_type = models.CharField(max_length=50, db_index=True)

    title = models.CharField(max_length=200)
    message = models.CharField(max_length=500)

    data = models.JSONField(
        default=dict, blank=True,
        help_text="Extra structured payload for the event, e.g. {\"document_id\": 12, \"share_id\": 7}.",
    )

    action_url = models.CharField(
        max_length=500, blank=True, default="",
        help_text='Where the "Open" action navigates, e.g. a document URL. Computed at creation time via reverse() - never trusted as an authorization check by itself, the target view re-validates access.',
    )

    is_read = models.BooleanField(default=False, db_index=True)
    read_at = models.DateTimeField(null=True, blank=True)

    email_sent = models.BooleanField(default=False)
    email_sent_at = models.DateTimeField(null=True, blank=True)
    email_error = models.CharField(
        max_length=255, blank=True, default="",
        help_text="Sanitized failure reason only (exception class name, not a raw traceback) - mirrors ErrorGroup's no-raw-stack-trace contract.",
    )

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["recipient", "is_read", "-created_at"])]

    @property
    def category(self):
        """First segment of notification_type, e.g. "document" - derived, not stored, same pattern as Permission.namespace."""
        return self.notification_type.split(".")[0]

    def __str__(self):
        return f"{self.notification_type} -> {self.recipient.username}"


class NotificationPreference(models.Model):
    """
    Per-user email-notification opt-outs. In-app notifications are never
    disableable (the inbox is always the full record). "account" and
    "security" category notifications (password changed, new sign-in,
    email verified, ...) are never offered as toggleable and always
    email regardless of this row - see
    RAG.services.notification_service.ALWAYS_EMAIL_CATEGORIES.
    """

    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="notification_preferences")

    disabled_email_categories = models.JSONField(
        default=list, blank=True,
        help_text="Category namespaces (e.g. \"document\", \"ai_task\") the user has opted out of *email* delivery for.",
    )

    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Notification preferences for {self.user.username}"


# ============================================================
# Organizations / Multi-Tenancy
# ============================================================
#
# A deliberately separate system from the platform RBAC above (Role/
# Permission/UserRole), not a generalization of it - a platform Admin
# and an organization's Owner/Admin are different concepts that happen
# to look structurally similar. See RAG/services/org_permission_service.py
# for the read/authorization API (mirrors permission_service.py's
# shape) and that module's docstring for the full rationale.
#
# Every existing user-owned resource (Document, QueryLog, Entity, ...)
# stays exactly as it is today - this section adds the tenancy
# primitives only. Retrofitting individual resources to become
# organization-aware (an additive, nullable `organization` FK each) is
# deliberately a separate, later change per resource, not part of this
# migration - see the architecture note this was designed against.
#
# There is no separate "Super Admin" role for organization oversight:
# the platform Admin role already has every permission by design
# (Role.has_permission's bypass above), so platform-wide organization
# management is expressed as new "organizations.*" permissions
# (seed_rbac.py) that Admin's "__all__" grant already covers, rather
# than inventing a second top-tier platform role alongside the one
# that was deliberately removed (see seed_rbac.py's docstring on the
# prior "super_admin" role).

ORG_ROLE_OWNER = "org_owner"
ORG_ROLE_MEMBER = "member"

# Retired 2026-09-06: organization roles collapsed from four tiers
# (Owner/Admin/Manager/Member) to two (Owner/Member) - product decision
# that an organization's Owner holds full control (RBAC, settings,
# billing, analytics, everything) and every other member gets none of
# that surface at all, full stop. These two string values are kept
# only so migration 00XX_collapse_org_roles_to_owner_member can
# reference the values it's migrating away FROM; nothing else should
# import or compare against them - OrganizationMembership.Role no
# longer offers them as choices, so no new row can ever get one.
_RETIRED_ORG_ROLE_ADMIN = "org_admin"
_RETIRED_ORG_ROLE_MANAGER = "manager"


class OrganizationType(models.Model):
    """
    An extensible organization category (Company, Startup, Enterprise,
    University, ...) - a table, not a hardcoded choices list, so a new
    type never needs a migration or code change. Seeded by the
    seed_organization_types management command; Admins can add more via
    Django Admin.
    """

    slug = models.SlugField(max_length=50, unique=True)
    name = models.CharField(max_length=100)
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return self.name


class Organization(models.Model):
    """
    A tenant. Membership (who belongs, with what role) lives on
    OrganizationMembership below, not here - deliberately no direct
    `owner` FK, since "who owns this org" is always derivable as
    `memberships.filter(role=ORG_ROLE_OWNER)` and a separate FK would
    just be a second copy of that fact that could drift out of sync.
    """

    class Status(models.TextChoices):
        ACTIVE = "active", "Active"
        SUSPENDED = "suspended", "Suspended"
        PENDING_DELETION = "pending_deletion", "Pending Deletion"

    name = models.CharField(max_length=150)

    slug = models.SlugField(
        max_length=160,
        unique=True,
        help_text="URL-safe public identifier (e.g. /app/org/<slug>/...) - never expose the numeric id in a user-facing URL.",
    )

    org_type = models.ForeignKey(
        OrganizationType, on_delete=models.PROTECT, related_name="organizations"
    )

    description = models.TextField(blank=True, default="")
    logo = models.ImageField(upload_to="organizations/logos/", blank=True, null=True)
    website = models.URLField(blank=True, default="")
    industry = models.CharField(max_length=100, blank=True, default="")
    contact_email = models.EmailField(blank=True, default="")
    contact_phone = models.CharField(max_length=30, blank=True, default="")

    class Size(models.TextChoices):
        SIZE_1_10 = "1-10", "1-10 employees"
        SIZE_11_50 = "11-50", "11-50 employees"
        SIZE_51_200 = "51-200", "51-200 employees"
        SIZE_201_500 = "201-500", "201-500 employees"
        SIZE_500_PLUS = "500+", "500+ employees"

    size = models.CharField(
        max_length=10, choices=Size.choices, blank=True, default="",
        help_text="Self-reported company size band, collected during company signup - purely informational, no behavior branches on it.",
    )

    status = models.CharField(
        max_length=20, choices=Status.choices, default=Status.ACTIVE, db_index=True
    )

    created_by = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, blank=True, related_name="organizations_created"
    )

    ai_credits_balance = models.PositiveIntegerField(
        null=True, blank=True, default=None,
        help_text="A spendable AI usage balance - the organization's SOLE spend-metering "
                   "currency (see billing_service.py's module docstring); the Plan's own "
                   "max_queries_per_month/max_ai_task_runs_per_month are informational quota "
                   "display only, not a second enforcement gate. See "
                   "billing_service.check_ai_credits()/deduct_ai_credits(). NULL (every "
                   "organization's default, until an Owner ever tops up) means unlimited, the same "
                   "'absence means today's behavior, nothing breaks until someone opts in' "
                   "convention Subscription's own absence already means for the Plan-based caps - a "
                   "brand-new organization must never be silently blocked from AI features it never "
                   "asked to meter. Only an Owner can add credits (billing_service.add_ai_credits()), "
                   "which is the one place this ever becomes a real (non-NULL) number; every "
                   "addition/deduction from that point on is recorded in AICreditTransaction below, "
                   "never inferred from this running total alone.",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    members = models.ManyToManyField(
        User,
        through="OrganizationMembership",
        through_fields=("organization", "user"),
        related_name="organizations",
    )

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return self.name


class OrganizationMembership(models.Model):
    """
    The User <-> Organization association, carrying that user's role
    and status WITHIN this one organization - a user can hold different
    roles in different organizations, and this table (not a field on
    User) is what makes that possible. `role` is a fixed two-tier
    CharField rather than a dynamic Role FK (unlike platform RBAC):
    Owner holds full control of the organization (RBAC, settings,
    billing, members, complete analytics - everything), and Member
    gets none of that surface at all - no Admin/Manager middle tier
    (collapsed away 2026-09-06; see org_permission_service.py's module
    docstring for the migration path to per-org custom roles if a
    middle tier is ever needed again). See
    org_permission_service.ORG_ROLE_PERMISSIONS for what each tier
    actually grants.

    Besides role/status, this is also where an Owner's two per-member
    controls live: `disabled_features` (which Plan-included feature
    pages this member can reach) and `max_queries_per_month`/
    `max_ai_task_runs_per_month` (how much of the organization's usage
    THIS member may consume per billing period, a count-based ceiling
    kept deliberately separate from the org-wide AI credit balance -
    see those fields' own help_text and billing_service.py's module
    docstring).
    """

    class Role(models.TextChoices):
        OWNER = ORG_ROLE_OWNER, "Owner"
        MEMBER = ORG_ROLE_MEMBER, "Member"

    class Status(models.TextChoices):
        ACTIVE = "active", "Active"
        SUSPENDED = "suspended", "Suspended"

    organization = models.ForeignKey(
        Organization, on_delete=models.CASCADE, related_name="memberships"
    )

    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="organization_memberships"
    )

    role = models.CharField(max_length=20, choices=Role.choices, default=Role.MEMBER)

    status = models.CharField(
        max_length=20, choices=Status.choices, default=Status.ACTIVE, db_index=True
    )

    invited_by = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, blank=True, related_name="+"
    )

    joined_at = models.DateTimeField(auto_now_add=True)

    disabled_features = models.JSONField(
        default=list, blank=True,
        help_text="Feature codes (RAG.models.FEATURE_CODES) this specific member is blocked "
                   "from, even though the organization's Plan includes them - Owner-only, set "
                   "via org_member_feature_service.set_member_disabled_features(). Empty = no "
                   "member-specific restriction, the same absence-means-allowed convention "
                   "Plan.included_features and Subscription's own absence both use. Never "
                   "applies to an Owner (see set_member_disabled_features()'s guard).",
    )

    max_queries_per_month = models.PositiveIntegerField(
        null=True, blank=True, default=None,
        help_text="Owner-set sub-quota on how many questions THIS member may ask per billing "
                   "period - a SEPARATE, count-based ceiling from the organization's own AI "
                   "credit balance (Organization.ai_credits_balance, the org-wide spend-metering "
                   "currency), enforced independently of it. Always bounded by (never exceeding) "
                   "the org's active Plan's own max_queries_per_month when one is set - see "
                   "org_member_limits_service.set_member_usage_limits()'s validation. NULL "
                   "(default) means no member-specific cap, the same absence-means-unrestricted "
                   "convention disabled_features above already uses. Counted against QueryLog "
                   "rows scoped to this user (billing_service.get_member_usage()) and enforced "
                   "as a hard block by billing_service.check_member_query_limit(). Never applies "
                   "to an Owner (see set_member_usage_limits()'s guard).",
    )

    max_ai_task_runs_per_month = models.PositiveIntegerField(
        null=True, blank=True, default=None,
        help_text="Same per-member sub-quota as max_queries_per_month above, counted against "
                   "AITaskRun rows instead and enforced by "
                   "billing_service.check_member_ai_task_limit().",
    )

    class Meta:
        unique_together = ("organization", "user")
        indexes = [
            models.Index(fields=["user", "status"]),
            models.Index(fields=["organization", "role"]),
        ]

    def __str__(self):
        return f"{self.user} @ {self.organization} ({self.role})"


class OrganizationInvitation(models.Model):
    """
    A pending invitation for someone (who may not have an account yet)
    to join an organization at a specific role. Mirrors DocumentShare's
    "invited_email is the pending state" pattern: this row exists
    independently of whether `email` has a matching User yet -
    org_invitation_service.accept_invitation() is the only place a
    PENDING row ever changes status, and it requires the accepting
    User's own email to match `email` exactly (see that function's
    docstring), so a token can never be redeemed into an account it
    wasn't actually issued to.

    `token` is the single-use credential - opaque
    (secrets.token_urlsafe, never the row's own id/pk) and only ever
    meaningful while status == PENDING; accept/revoke both flip status
    away from PENDING as their very first write, so a token cannot be
    replayed after either.
    """

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        ACCEPTED = "accepted", "Accepted"
        EXPIRED = "expired", "Expired"
        REVOKED = "revoked", "Revoked"

    organization = models.ForeignKey(
        Organization, on_delete=models.CASCADE, related_name="invitations"
    )

    email = models.EmailField(db_index=True)

    role = models.CharField(
        max_length=20, choices=OrganizationMembership.Role.choices, default=OrganizationMembership.Role.MEMBER
    )

    token = models.CharField(max_length=64, unique=True, db_index=True)

    status = models.CharField(
        max_length=20, choices=Status.choices, default=Status.PENDING, db_index=True
    )

    invited_by = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, blank=True, related_name="organization_invitations_sent"
    )

    accepted_by = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, blank=True, related_name="organization_invitations_accepted"
    )

    expires_at = models.DateTimeField()

    created_at = models.DateTimeField(auto_now_add=True)
    accepted_at = models.DateTimeField(null=True, blank=True)
    revoked_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["organization", "status"]),
            models.Index(fields=["email", "status"]),
        ]
        constraints = [
            # Only one PENDING invitation per (organization, email) at
            # a time - same "scoped/partial uniqueness" technique
            # DocumentShare.Meta uses for its own pending-invite case,
            # for the same reason: a plain unique_together would also
            # collide every ACCEPTED/EXPIRED/REVOKED historical row for
            # that same address, which is exactly the history a re-
            # invite after expiry needs to be able to create fresh.
            models.UniqueConstraint(
                fields=["organization", "email"],
                condition=models.Q(status="pending"),
                name="orginvitation_unique_pending_per_org_email",
            ),
        ]

    def __str__(self):
        return f"{self.email} -> {self.organization} ({self.role}, {self.status})"


class OrganizationAuditLog(models.Model):
    """
    Organization-scoped audit trail - deliberately a SEPARATE table
    from the platform-wide ActivityLog above, not that model with an
    optional `organization` FK bolted on. An org Owner/Admin must see
    only their own organization's events; a separate table makes "an
    org-scoped query cannot return another org's rows" a schema-level
    fact enforced by the FK itself, not something every call site has
    to remember to filter for. Written by
    RAG.services.org_audit_log_service.log_org_activity() - membership
    changes, role changes, invitations, and (once retrofitted)
    organization-owned resource events.
    """

    organization = models.ForeignKey(
        Organization, on_delete=models.CASCADE, related_name="audit_logs"
    )

    actor = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, blank=True, related_name="org_audit_logs"
    )

    action = models.CharField(
        max_length=50, db_index=True,
        help_text='Namespaced event, e.g. "member.invited", "member.role_changed" - same convention as ActivityLog.action.',
    )

    description = models.CharField(max_length=255)

    metadata = models.JSONField(default=dict, blank=True)

    ip_address = models.GenericIPAddressField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["organization", "-created_at"])]

    def __str__(self):
        return f"[{self.organization}] {self.action}: {self.description}"


# ============================================================
# Billing / Plans / Usage Limits
# ============================================================
#
# Internal bookkeeping only - no real payment processor, no card
# charging. A Super Admin manually assigns a Plan to an Organization.
# Two independent enforcement axes, deliberately not merged into one:
#   - AI credits (Organization.ai_credits_balance) are the organization's
#     SOLE spend-metering currency - every question/AI Task run costs
#     credits, and running out is the one hard org-wide block
#     (billing_service.check_ai_credits()). Plan.max_queries_per_month/
#     max_ai_task_runs_per_month are informational quota display only
#     now (see Plan's own field help_text below) - they no longer raise
#     on their own, which is what used to let an org get blocked by a
#     count cap while sitting on an untouched credit balance, or vice
#     versa, two gates metering the same action.
#   - Per-member quantity caps (OrganizationMembership.
#     max_queries_per_month/max_ai_task_runs_per_month) are a SEPARATE,
#     Owner-set ceiling on one specific member's usage, count-based (not
#     credits), and enforced independently of the org-wide credit
#     balance - see billing_service.check_member_query_limit()/
#     check_member_ai_task_limit().
# No Subscription row for an organization means unlimited - the same
# "absence means today's behavior" rollout convention the Departments
# models above use.

# Feature-page codes gated by Plan.included_features / OrganizationMembership
# .disabled_features (org_member_feature_service.py). Deliberately a separate
# list from the platform pages.* Permission codenames (seed_rbac.py) even
# though the names line up 1:1 - this gates a structurally different
# (Plan/member) system, not platform RBAC, and the two must not be conflated.
FEATURE_CODES = ["documents", "knowledge_base", "ask_ai", "ai_tasks", "analytics", "reports"]

FEATURE_LABELS = {
    "documents": "Documents",
    "knowledge_base": "Knowledge Base",
    "ask_ai": "Ask AI",
    "ai_tasks": "AI Tasks",
    "analytics": "Analytics",
    "reports": "Reports",
}


class Plan(models.Model):
    """
    Super-Admin-managed only - nothing in the Owner-facing (or Personal-
    Workspace-facing) API can create/edit a Plan or exceed what one
    grants (see billing_views.py: every Plan-writing endpoint is gated
    on the platform "billing.manage_plans" permission, never anything
    org-scoped). NULL on any max_* field means "unlimited" for that
    dimension. `plan_type` is the one field that separates the Company
    and Personal Workspace plan catalogs - both live in this same table
    (identical shape: price/credits/features/limits) rather than two
    parallel models, since a Super Admin manages both through one CRUD
    surface (AdminBillingPlans.jsx's tabs just filter on this field).
    """

    class PlanType(models.TextChoices):
        COMPANY = "company", "Company"
        PERSONAL = "personal", "Personal"

    class BillingInterval(models.TextChoices):
        MONTHLY = "monthly", "Monthly"
        YEARLY = "yearly", "Yearly"

    name = models.CharField(max_length=100, unique=True)
    slug = models.SlugField(max_length=110, unique=True)
    description = models.CharField(max_length=255, blank=True, default="")

    plan_type = models.CharField(max_length=10, choices=PlanType.choices, default=PlanType.COMPANY)

    price = models.DecimalField(
        max_digits=10, decimal_places=2, null=True, blank=True,
        help_text="NULL or 0 = free plan. Internal bookkeeping only - no real payment "
                   "processor charges this (see billing_service.py's module docstring).",
    )
    currency = models.CharField(max_length=3, default="USD")
    billing_interval = models.CharField(max_length=10, choices=BillingInterval.choices, default=BillingInterval.MONTHLY)

    max_queries_per_month = models.PositiveIntegerField(
        null=True, blank=True,
        help_text="Informational monthly quota, shown alongside usage for reference - NOT an "
                   "independent enforcement gate. AI credits (included_credits below) are the "
                   "organization's sole spend-metering currency (see billing_service.py's module "
                   "docstring for why these two axes were merged into one); an Owner can still "
                   "narrow an individual member below whatever this number implies via "
                   "OrganizationMembership.max_queries_per_month, which IS a real hard block.",
    )
    max_ai_task_runs_per_month = models.PositiveIntegerField(
        null=True, blank=True,
        help_text="Same informational-only role as max_queries_per_month above, for AI Task runs.",
    )
    max_storage_bytes = models.BigIntegerField(null=True, blank=True)
    max_seats = models.PositiveIntegerField(null=True, blank=True)

    included_credits = models.PositiveIntegerField(
        default=0,
        help_text="AI credits this Plan grants each billing period - refilled to this exact "
                   "amount on every period rollover (see billing_service._advance_period()), "
                   "not accumulated. 0 = this Plan includes no credits.",
    )
    allow_credit_purchase = models.BooleanField(
        default=True,
        help_text="Whether an Owner on this Plan may buy extra credits as an "
                   "add-on (billing_service.add_ai_credits()) on top "
                   "of included_credits.",
    )

    included_features = models.JSONField(
        default=list, blank=True,
        help_text="Feature codes (FEATURE_CODES) this Plan grants. Empty list = every feature "
                   "included (unrestricted) - same 'absence means unlimited' convention as "
                   "max_queries_per_month=None, so every Plan created before this field existed "
                   "keeps granting full access with zero admin action required.",
    )

    is_active = models.BooleanField(
        default=True,
        help_text="Deactivating hides a Plan from future assignment without deleting it - "
                   "existing Subscriptions referencing it (PROTECT below) keep working.",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["plan_type", "name"]

    def __str__(self):
        return self.name


class Subscription(models.Model):
    """
    One active Plan assignment per Organization OR per personal-
    Workspace User - exactly one of `organization`/`user` is set,
    enforced by the CheckConstraint below (the same "organization=None
    means Personal Workspace" convention this codebase uses everywhere
    else, just made explicit at the DB level here since this one model
    now has to represent both scopes). No history of past plans kept
    here (OrganizationAuditLog's "billing.plan_assigned" entries are
    the history); deleting this row (not a status flag) is how a Super
    Admin reverts an organization/user to unlimited - see
    billing_service.get_active_subscription()'s "no row = unlimited"
    contract. A *request* for a new Plan that hasn't been approved yet
    is a separate PlanChangeRequest row, never a Subscription - this
    row only ever represents an already-active grant.
    """

    # Both nullable OneToOneFields (not plain ForeignKeys): at most one
    # Subscription per organization and at most one per user, AND the
    # singular `organization.subscription`/`user.personal_subscription`
    # reverse accessor (used by select_related("subscription__plan") in
    # billing_views.py/admin_system_overview_views.py) both depend on
    # this being a real OneToOneField, not just a uniqueness constraint
    # on a ForeignKey - Django's select_related() can only traverse the
    # reverse side of a genuine O2O, never a reverse FK. Postgres allows
    # multiple NULLs in a unique index, so nullable O2O naturally
    # supports "many rows with organization=NULL" (one per distinct
    # user) while still enforcing "at most one row per non-null org".
    organization = models.OneToOneField(Organization, on_delete=models.CASCADE, null=True, blank=True, related_name="subscription")
    user = models.OneToOneField(User, on_delete=models.CASCADE, null=True, blank=True, related_name="personal_subscription")
    plan = models.ForeignKey(Plan, on_delete=models.PROTECT, related_name="subscriptions")

    current_period_start = models.DateTimeField()
    current_period_end = models.DateTimeField()

    assigned_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name="+")
    assigned_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [models.Index(fields=["organization", "current_period_end"])]
        constraints = [
            models.CheckConstraint(
                condition=(
                    models.Q(organization__isnull=False, user__isnull=True)
                    | models.Q(organization__isnull=True, user__isnull=False)
                ),
                name="subscription_exactly_one_of_organization_or_user",
            ),
        ]

    def __str__(self):
        return f"{self.organization or self.user} -> {self.plan}"


class PlanChangeRequest(models.Model):
    """
    An Owner asking to switch to a different Plan - stays Pending
    until a Platform Admin approves or rejects it (billing_service.
    approve_plan_request()/reject_plan_request()). Approval is what
    actually creates/updates the real Subscription row (via the
    existing assign_plan()) - this model never grants anything on its
    own, it's purely the request/audit trail in front of that. This is
    the ONLY way an existing organization's plan ever changes after
    creation - there is deliberately no admin-initiated instant
    override that skips this queue (see billing_views.py's module
    docstring). A brand-new organization's very first Subscription is
    the one exception, created automatically with the built-in Free
    plan at organization creation (organization_service.
    create_organization()), never through this request model at all.
    """

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        APPROVED = "approved", "Approved"
        REJECTED = "rejected", "Rejected"

    organization = models.ForeignKey(Organization, on_delete=models.CASCADE, null=True, blank=True, related_name="plan_requests")
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True, related_name="personal_plan_requests")
    requested_plan = models.ForeignKey(Plan, on_delete=models.PROTECT, related_name="+")

    status = models.CharField(max_length=10, choices=Status.choices, default=Status.PENDING)
    requested_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name="+")
    reviewed_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name="+")
    reviewed_at = models.DateTimeField(null=True, blank=True)
    admin_note = models.CharField(max_length=255, blank=True, default="")

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["-created_at"]
        constraints = [
            models.CheckConstraint(
                condition=(
                    models.Q(organization__isnull=False, user__isnull=True)
                    | models.Q(organization__isnull=True, user__isnull=False)
                ),
                name="plan_request_exactly_one_of_organization_or_user",
            ),
        ]
        indexes = [models.Index(fields=["status", "created_at"])]

    def __str__(self):
        return f"{self.organization or self.user} -> {self.requested_plan} ({self.status})"


class AICreditTransaction(models.Model):
    """
    One line of an organization's AI credit ledger - every top-up
    (positive `amount`, either an Owner self-serve purchase via
    billing_service.add_ai_credits(), or an automatic Plan-period
    refill) and every spend (negative `amount`, deducted automatically
    per question asked / AI Task run via billing_service.
    deduct_ai_credits() - see that module for the cost schedule).
    `balance_after` is a
    denormalized snapshot of the balance at the moment this row was
    written - kept so the ledger reads back as a real running-balance
    history without recomputing a sum over every prior row, the same
    "correct once, cheap to read forever" trade-off QueryLog/AITaskRun
    already make for their own tables. Organization.ai_credits_balance/
    UserProfile.ai_credits_balance themselves remain the single source
    of truth for "how many credits right now" - this table is the
    audit trail behind that number, not a second place it's computed
    from. Exactly one of `organization`/`user` is set, same convention
    as Subscription above - Company and Personal credit pools are
    completely separate ledgers, never mixed or transferable.
    """

    organization = models.ForeignKey(Organization, on_delete=models.CASCADE, null=True, blank=True, related_name="ai_credit_transactions")
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True, related_name="personal_ai_credit_transactions")
    amount = models.IntegerField(help_text="Positive for a top-up, negative for a spend.")
    balance_after = models.PositiveIntegerField()
    reason = models.CharField(max_length=100, help_text='e.g. "Question asked", "AI Task run", "Credits added by Owner", "Plan period refill".')
    actor = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name="+")
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["-created_at"]
        constraints = [
            models.CheckConstraint(
                condition=(
                    models.Q(organization__isnull=False, user__isnull=True)
                    | models.Q(organization__isnull=True, user__isnull=False)
                ),
                name="ai_credit_transaction_exactly_one_of_organization_or_user",
            ),
        ]

    def __str__(self):
        return f"{self.organization or self.user} {self.amount:+d} ({self.reason})"
