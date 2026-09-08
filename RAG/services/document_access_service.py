"""
Document Access Control - single source of truth for "can this user see
this document."

The rest of the app treats a document as accessible if the requesting
user owns it, it's in the Admin-managed Organization Library, or it's
been explicitly shared with them (directly, or via a Role they hold).
Every retrieval function (vector/BM25/graph search) and every
Document-Center view that isn't strictly owner-only (preview, download,
favorites, the Select Documents dialog) goes through
get_accessible_document_ids() rather than re-deriving this set itself,
so the access boundary only has one place to get right.

Fails closed: a missing/unauthenticated user always yields an empty
set, never "everyone" - mirrors the fail-closed contract
retrieval_service.py/bm25_service.py/graph_retrieval_service.py already
use for `user=None`.

MULTI-TENANCY: every function here now takes an optional `organization`
keyword. Omitted (the default), a call behaves EXACTLY as it always
has - personal-workspace scope. Passed a real Organization, the SAME
owned/library/shared shape applies, just scoped to that tenant instead
of to Personal Workspace - organization membership (verified by the
caller, not here) is what lets someone act inside a tenant at all, but
it is NOT itself blanket visibility into every document that tenant
owns (a real, explicitly product-confirmed decision: a member's
upload defaults to private to them within their own company too,
exactly like Personal Workspace already worked - is_org_library is
what actually publishes it company-wide, in either workspace). This
function does NOT re-verify the caller is actually a member of
`organization`; every caller MUST have already done that via
org_permission_service.get_org_membership()/resolve_request_organization()
before reaching here, exactly like this function already trusts its
caller to have authenticated `user` upstream. Getting this trust
boundary right is the one thing this module exists to centralize -
see the module docstring above.
"""

from django.db.models import Q

from ..models import Document
from .permission_service import get_user_role


def get_accessible_document_ids(user, organization=None):
    """
    Every Document id `user` may view, in the given workspace: owned
    by them ∪ that workspace's Organization Library ∪ shared directly
    with them ∪ shared with a Role they currently hold - all
    restricted to `organization` (`None` for Personal Workspace, a
    real Organization for that tenant). One shared shape for both
    workspaces, not a separate "org membership means see everything"
    branch - organization membership only grants ACTING inside a
    tenant, never blanket read access to every document it owns; see
    the module docstring for why.

    Returns a set, computed fresh on every call (no caching) so a
    revoked share, role change, or membership change takes effect on
    the very next query.
    """

    if user is None or not getattr(user, "is_authenticated", False):
        return set()

    scope = (
        Q(user=user, organization=organization)
        | Q(is_org_library=True, organization=organization)
        | Q(shares__shared_with_user=user, organization=organization)
    )

    role = get_user_role(user)
    if role is not None:
        scope |= Q(shares__shared_with_role=role, organization=organization)

    return set(Document.objects.filter(scope).values_list("id", flat=True))


def get_accessible_documents(user, organization=None):
    """The queryset counterpart of get_accessible_document_ids(), for views that need Document objects rather than bare ids."""

    ids = get_accessible_document_ids(user, organization=organization)
    return Document.objects.filter(id__in=ids) if ids else Document.objects.none()


def can_view_document(user, document):
    """
    Scopes itself to `document.organization` automatically (personal
    vs. that specific tenant) rather than taking an `organization`
    kwarg like the functions above - a single document object is
    already unambiguous about which workspace it belongs to, so there
    is no separate "which scope am I asking about" the caller could
    get wrong here.
    """

    return document is not None and document.id in get_accessible_document_ids(user, organization=document.organization)


def can_edit_document(user, document):
    """
    Owner-only, always - sharing and Organization Library membership
    grant view/download, never delete, re-versioning, or share
    management. There is no permission that widens this. For an
    organization-owned document this still means its original
    uploader, not "any org Owner/Admin" - a future
    org-role-can-manage-any-org-document policy is a deliberate,
    separate decision, not a side effect of this function.
    """

    return user is not None and document is not None and document.user_id == user.id
