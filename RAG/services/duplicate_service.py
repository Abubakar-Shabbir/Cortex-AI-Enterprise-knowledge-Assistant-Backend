import hashlib

from ..models import Document


def calculate_file_hash(file):
    """
    Calculate SHA-256 hash
    of uploaded file.
    """

    sha256 = hashlib.sha256()

    for chunk in file.chunks():
        sha256.update(chunk)

    file.seek(0)

    return sha256.hexdigest()


def check_duplicate(user, file, organization=None):
    """
    Check whether the uploaded document already exists - scoped to
    `organization` (or, when None, to the user's Personal Workspace
    specifically via organization__isnull=True) so uploading the exact
    same file to two different workspaces is never blocked as a false
    "already uploaded" - Personal Workspace and each organization keep
    independent duplicate history, matching how document_access_service
    already treats them as non-overlapping scopes everywhere else.
    """

    file_hash = calculate_file_hash(file)

    exists = Document.objects.filter(
        user=user,
        file_hash=file_hash,
        organization=organization,
    ).exists()

    return exists, file_hash