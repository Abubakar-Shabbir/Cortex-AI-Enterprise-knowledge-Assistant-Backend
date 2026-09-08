"""
Knowledge Graph construction.

Turns LLM-extracted entities/relationships (graph_extraction_service)
into Entity, EntityMention and Relationship rows, scoped per user.
Building the graph for a chunk is idempotent: re-running it for the
same chunk creates no duplicate entities, mentions, or edges.
"""

import logging

from django.db import transaction
from django.db.models import F

from ..models import Entity, EntityMention, Relationship
from .graph_extraction_service import extract_graph, normalize_entity_key

logger = logging.getLogger(__name__)


def _get_or_create_entity(user, organization, name: str, entity_type: str) -> Entity:

    key = normalize_entity_key(name)

    entity, _ = Entity.objects.get_or_create(
        user=user,
        organization=organization,
        name=key,
        entity_type=entity_type,
        defaults={"display_name": name},
    )

    return entity


def build_graph_for_chunk(chunk, user) -> None:
    """
    Extract entities/relationships from a single DocumentChunk and
    merge them into the user's knowledge graph.

    `organization` is never a parameter here - it's always derived
    from `chunk.document.organization`, exactly like
    document_access_service.can_view_document() derives it from the
    document rather than trusting a caller-supplied value: a chunk's
    document is already unambiguous about which workspace its
    extracted entities/relationships belong to (Entity/Relationship's
    `(user, organization, name, entity_type)` uniqueness - see
    models.py - so a Personal Workspace "Acme Corp" entity and an
    organization's own "Acme Corp" entity for the same uploader never
    collapse into one row and conflate their mention counts, even
    though EntityMention -> chunk -> document is still the only real
    access-control boundary graph_retrieval_service.py/
    knowledge_service.py ever check).

    Safe to call repeatedly for the same chunk. Never raises - a
    failed or empty extraction just means no graph enrichment for
    this chunk; the chunk and its embedding are unaffected.
    """

    try:
        result = extract_graph(chunk.content)
    except Exception:
        logger.exception(
            "Graph construction: extraction failed for chunk %s", chunk.pk
        )
        return

    if not result.entities:
        return

    _persist_graph(chunk, user, chunk.document.organization, result)


@transaction.atomic
def _persist_graph(chunk, user, organization, result) -> None:
    """
    Write the already-extracted entities/relationships to the
    database. Split out from build_graph_for_chunk() so the atomic
    block only spans DB writes, not the (much slower) LLM call.
    """

    entities_by_key = {}

    for extracted in result.entities:

        entity = _get_or_create_entity(user, organization, extracted.name, extracted.type)
        entities_by_key[normalize_entity_key(extracted.name)] = entity

        _, mention_created = EntityMention.objects.get_or_create(
            entity=entity,
            chunk=chunk,
        )

        if mention_created:
            Entity.objects.filter(pk=entity.pk).update(
                mention_count=F("mention_count") + 1
            )

    for rel in result.relationships:

        source = entities_by_key.get(normalize_entity_key(rel.source))
        target = entities_by_key.get(normalize_entity_key(rel.target))

        if not source or not target or source.pk == target.pk:
            continue

        relationship, created = Relationship.objects.get_or_create(
            user=user,
            organization=organization,
            source=source,
            target=target,
            relation_type=rel.relation,
            defaults={"context": chunk.content[:500]},
        )

        if not created:
            Relationship.objects.filter(pk=relationship.pk).update(
                weight=F("weight") + 1
            )

    logger.info(
        "Graph construction: chunk %s -> %d entities, %d relationships",
        chunk.pk, len(result.entities), len(result.relationships),
    )
