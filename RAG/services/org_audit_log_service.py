"""
Organization Audit Log - writes to OrganizationAuditLog, the tenant-
scoped counterpart to activity_log_service.log_activity(). Deliberately
NOT the same function with an extra parameter: the two write to
different tables on purpose (see OrganizationAuditLog's docstring), so
keeping them as separate, small functions makes it obvious at every
call site which trail an event is going into, rather than relying on
an argument to route it correctly.
"""

import logging

from .geolocation_service import get_client_ip

logger = logging.getLogger(__name__)


def log_org_activity(organization, actor, action, description, metadata=None, request=None):
    """
    Write one OrganizationAuditLog row. `request`, when given, is only
    used for its IP address - unlike activity_log_service.log_activity,
    there is no geolocation resolution here (an org audit trail cares
    about who-did-what-when within the tenant, not where in the world
    the request came from), so this never makes the external
    geolocation HTTP call activity_log_service sometimes does.

    Never raises - the same never-raise contract this codebase's other
    best-effort side-effect writers use (graph_extraction_service.
    extract_graph(), notification_service.create_notification()'s email
    dispatch, ...). A failed audit-log INSERT (a bad metadata value, a
    DB hiccup) must never fail the primary action it's recording (e.g.
    inviting a member, deleting a document) - the primary write already
    happened by the time this is called, so raising here would only
    turn a successful action into a misleading 500 for the caller.
    """

    from ..models import OrganizationAuditLog

    try:
        OrganizationAuditLog.objects.create(
            organization=organization,
            actor=actor,
            action=action[:50],
            description=description[:255],
            metadata=metadata or {},
            ip_address=get_client_ip(request) if request else None,
        )
    except Exception:
        logger.exception(
            "log_org_activity: failed to record '%s' for organization %s",
            action, getattr(organization, "id", organization),
        )
