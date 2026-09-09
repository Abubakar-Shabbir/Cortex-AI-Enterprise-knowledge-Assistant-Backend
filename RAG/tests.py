import json
import unittest
from datetime import date
from unittest.mock import MagicMock, patch

from django.contrib.auth.models import User
from django.core.cache import cache
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import RequestFactory, TestCase, TransactionTestCase, override_settings

from .models import AITaskRun, AITaskRunDocument, Document, DocumentChunk, DocumentShare, Entity, EntityMention, Notification, Organization, OrganizationInvitation, OrganizationMembership, OrganizationType, Permission, Plan, QueryLog, Relationship, Role, Subscription, UserProfile, UserRole
from .services import ai_tasks_engine_service as ai_tasks_engine
from .services import dynamic_topk_service as dynamic_topk
from .services import graph_extraction_service as extraction
from .services import graph_retrieval_service as graph_retrieval
from .services import hyde_service as hyde
from .services import llm_client
from .services import query_expansion_service as expansion
from . import tasks
from .services import citation_service
from .services import context_compression_service as compression
from .services import health_service
from .services import llm_service
from .services import prompt_templates
from .services import query_service
from .services import query_transform_service as transform
from .services import reranker_service as reranker
from .services import retrieval_service as retrieval
from .services.graph_extraction_service import (
    ExtractedEntity,
    ExtractedRelationship,
    GraphExtractionResult,
)
from .services.graph_service import build_graph_for_chunk
from .services.multi_query_service import _reciprocal_rank_fusion, multi_query_search
from .services.retrieval_filters import RetrievalFilters, apply_document_filters


class EntityNormalizationTests(unittest.TestCase):
    """
    Pure normalization helpers - no DB, no LLM calls.
    """

    def test_normalize_entity_name_collapses_whitespace(self):
        self.assertEqual(
            extraction.normalize_entity_name("  Acme   Corp\n"),
            "Acme Corp",
        )

    def test_normalize_entity_key_is_case_insensitive(self):
        self.assertEqual(extraction.normalize_entity_key("Acme Corp"), "acme corp")
        self.assertEqual(extraction.normalize_entity_key("ACME CORP"), "acme corp")

    def test_normalize_entity_type_defaults_to_misc(self):
        self.assertEqual(extraction.normalize_entity_type(""), "MISC")
        self.assertEqual(extraction.normalize_entity_type(None), "MISC")

    def test_normalize_entity_type_uppercases_and_underscores(self):
        self.assertEqual(extraction.normalize_entity_type("job title"), "JOB_TITLE")

    def test_normalize_relation_type_defaults_to_related_to(self):
        self.assertEqual(extraction.normalize_relation_type(""), "RELATED_TO")

    def test_normalize_relation_type_uppercases_and_underscores(self):
        self.assertEqual(extraction.normalize_relation_type("works for"), "WORKS_FOR")


class GraphExtractionParsingTests(unittest.TestCase):
    """
    _parse_response() / extract_graph() - no live Gemini calls, the
    model is mocked so these run offline and deterministically.
    """

    def test_parse_response_valid_json(self):
        raw = (
            '{"entities": [{"name": "Ada Lovelace", "type": "person"}, '
            '{"name": "Analytical Engine", "type": "product"}], '
            '"relationships": [{"source": "Ada Lovelace", "relation": "designed for", '
            '"target": "Analytical Engine"}]}'
        )

        result = extraction._parse_response(raw)

        self.assertEqual(len(result.entities), 2)
        self.assertEqual(result.entities[0].name, "Ada Lovelace")
        self.assertEqual(result.entities[0].type, "PERSON")

        self.assertEqual(len(result.relationships), 1)
        self.assertEqual(result.relationships[0].relation, "DESIGNED_FOR")

    def test_parse_response_drops_relationships_with_unknown_entities(self):
        raw = (
            '{"entities": [{"name": "Ada Lovelace", "type": "person"}], '
            '"relationships": [{"source": "Ada Lovelace", "relation": "knows", '
            '"target": "Someone Never Extracted"}]}'
        )

        result = extraction._parse_response(raw)

        self.assertEqual(len(result.entities), 1)
        self.assertEqual(result.relationships, [])

    def test_parse_response_malformed_json_returns_empty_result(self):
        result = extraction._parse_response("not json at all")

        self.assertEqual(result.entities, [])
        self.assertEqual(result.relationships, [])

    def test_extract_graph_skips_short_text_without_calling_llm(self):
        with patch.object(extraction, "get_llm") as mock_get_llm:
            result = extraction.extract_graph("too short")

        mock_get_llm.assert_not_called()
        self.assertEqual(result.entities, [])

    def test_extract_graph_returns_empty_result_on_llm_failure(self):
        with patch.object(extraction, "get_llm", side_effect=RuntimeError("boom")):
            result = extraction.extract_graph(
                "Ada Lovelace worked with Charles Babbage on the Analytical Engine."
            )

        self.assertEqual(result.entities, [])
        self.assertEqual(result.relationships, [])

    def test_extract_graph_parses_mocked_llm_response(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = (
            '{"entities": [{"name": "Charles Babbage", "type": "person"}], '
            '"relationships": []}'
        )

        with patch.object(extraction, "get_llm", return_value=mock_llm):
            result = extraction.extract_graph(
                "Charles Babbage designed the Difference Engine."
            )

        self.assertEqual(len(result.entities), 1)
        self.assertEqual(result.entities[0].name, "Charles Babbage")


class GraphConstructionTests(TestCase):
    """
    build_graph_for_chunk() against a real (test) database, with the
    LLM extraction call mocked so no network/API key is required.
    """

    def setUp(self):
        self.user = User.objects.create_user(username="tester", password="pw")

        self.document = Document.objects.create(
            user=self.user,
            title="Test Document",
            file="documents/test.txt",
        )

        self.chunk = DocumentChunk.objects.create(
            document=self.document,
            content="Ada Lovelace worked with Charles Babbage.",
            chunk_number=0,
        )

    def _extraction_result(self):
        return GraphExtractionResult(
            entities=[
                ExtractedEntity(name="Ada Lovelace", type="PERSON"),
                ExtractedEntity(name="Charles Babbage", type="PERSON"),
            ],
            relationships=[
                ExtractedRelationship(
                    source="Ada Lovelace",
                    relation="WORKED_WITH",
                    target="Charles Babbage",
                ),
            ],
        )

    def test_build_graph_creates_entities_and_relationship(self):
        with patch(
            "RAG.services.graph_service.extract_graph",
            return_value=self._extraction_result(),
        ):
            build_graph_for_chunk(self.chunk, self.user)

        self.assertEqual(Entity.objects.filter(user=self.user).count(), 2)
        self.assertEqual(Relationship.objects.filter(user=self.user).count(), 1)
        self.assertEqual(EntityMention.objects.filter(entity__user=self.user).count(), 2)

        relationship = Relationship.objects.get(user=self.user)
        self.assertEqual(relationship.weight, 1)
        self.assertEqual(relationship.relation_type, "WORKED_WITH")

    def test_build_graph_is_idempotent_and_increments_weight(self):
        with patch(
            "RAG.services.graph_service.extract_graph",
            return_value=self._extraction_result(),
        ):
            build_graph_for_chunk(self.chunk, self.user)
            build_graph_for_chunk(self.chunk, self.user)

        # Same chunk processed twice: no duplicate entities/mentions,
        # but the relationship weight reflects the repeat extraction.
        self.assertEqual(Entity.objects.filter(user=self.user).count(), 2)
        self.assertEqual(EntityMention.objects.filter(entity__user=self.user).count(), 2)

        relationship = Relationship.objects.get(user=self.user)
        self.assertEqual(relationship.weight, 2)

    def test_build_graph_skips_self_loop_relationships(self):
        result = GraphExtractionResult(
            entities=[ExtractedEntity(name="Ada Lovelace", type="PERSON")],
            relationships=[
                ExtractedRelationship(
                    source="Ada Lovelace",
                    relation="KNOWS",
                    target="Ada Lovelace",
                ),
            ],
        )

        with patch("RAG.services.graph_service.extract_graph", return_value=result):
            build_graph_for_chunk(self.chunk, self.user)

        self.assertEqual(Relationship.objects.filter(user=self.user).count(), 0)

    def test_build_graph_noop_on_extraction_failure(self):
        with patch(
            "RAG.services.graph_service.extract_graph",
            side_effect=RuntimeError("boom"),
        ):
            build_graph_for_chunk(self.chunk, self.user)

        self.assertEqual(Entity.objects.filter(user=self.user).count(), 0)


class GraphRetrievalTests(TestCase):

    def setUp(self):
        self.user = User.objects.create_user(username="tester", password="pw")
        self.other_user = User.objects.create_user(username="other", password="pw")

        self.document = Document.objects.create(
            user=self.user,
            title="Test Document",
            file="documents/test.txt",
        )

        self.chunk = DocumentChunk.objects.create(
            document=self.document,
            content="Ada Lovelace pioneered early computing concepts.",
            chunk_number=0,
        )

        self.entity = Entity.objects.create(
            user=self.user,
            name="ada lovelace",
            display_name="Ada Lovelace",
            entity_type="PERSON",
            mention_count=1,
        )

        EntityMention.objects.create(entity=self.entity, chunk=self.chunk)

    def test_graph_search_returns_empty_without_user(self):
        self.assertEqual(graph_retrieval.graph_search("Who is Ada Lovelace?", None, 3), [])

    def test_graph_search_matches_entity_mentioned_in_question(self):
        results = graph_retrieval.graph_search("Who is Ada Lovelace?", self.user, 3)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["document"], "Test Document")
        self.assertEqual(results[0]["search_type"], "graph")

    def test_graph_search_is_scoped_to_user(self):
        results = graph_retrieval.graph_search("Who is Ada Lovelace?", self.other_user, 3)

        self.assertEqual(results, [])

    def test_graph_search_no_match_returns_empty(self):
        results = graph_retrieval.graph_search("What's the weather today?", self.user, 3)

        self.assertEqual(results, [])


class DescribeSearchMethodTests(unittest.TestCase):
    """
    Pure function - no DB required.
    """

    def test_no_results_keeps_original_default_label(self):
        self.assertEqual(query_service.describe_search_method([]), "Hybrid (Vector + BM25)")

    def test_vector_and_bm25_only_matches_original_label(self):
        chunks = [{"search_type": "vector"}, {"search_type": "bm25"}]
        self.assertEqual(query_service.describe_search_method(chunks), "Hybrid (Vector + BM25)")

    def test_graph_contribution_is_reflected_in_label(self):
        chunks = [{"search_type": "vector"}, {"search_type": "graph"}]
        self.assertEqual(
            query_service.describe_search_method(chunks),
            "Hybrid (Vector + Graph)",
        )

    def test_hyde_and_multi_query_labels(self):
        chunks = [{"search_type": "hyde"}, {"search_type": "multi_query"}]
        self.assertEqual(
            query_service.describe_search_method(chunks),
            "Hybrid (HyDE + Multi-query)",
        )


class CalculateConfidenceTests(unittest.TestCase):
    """
    Pure function - no DB required.
    """

    def test_hyde_scores_count_toward_confidence_like_vector(self):
        # A close HyDE match (small L2 distance) should read as high
        # confidence, the same as an equally close vector match.
        vector_confidence = query_service.calculate_confidence(
            [{"search_type": "vector", "score": 0.1}]
        )
        hyde_confidence = query_service.calculate_confidence(
            [{"search_type": "hyde", "score": 0.1}]
        )
        self.assertEqual(vector_confidence, hyde_confidence)

    def test_bm25_and_multi_query_only_fall_back_to_neutral(self):
        chunks = [
            {"search_type": "bm25", "score": 5.0},
            {"search_type": "multi_query", "score": 0.05},
        ]
        self.assertEqual(query_service.calculate_confidence(chunks), 40)

    def test_not_found_answer_forces_zero_confidence(self):
        chunks = [{"search_type": "vector", "score": 0.01}]
        confidence = query_service.calculate_confidence(
            chunks, answer=prompt_templates.NOT_FOUND_ANSWER
        )
        self.assertEqual(confidence, 0)

    def test_zero_citations_discounts_confidence(self):
        chunks = [{"search_type": "vector", "score": 0.1}]

        uncited = query_service.calculate_confidence(
            chunks, answer="An answer with no markers.", citation_count=0
        )
        cited = query_service.calculate_confidence(
            chunks, answer="An answer [1].", citation_count=1
        )

        self.assertLess(uncited, cited)

    def test_citation_count_none_skips_discount(self):
        # Backward compatibility: omitting citation_count entirely
        # (existing callers) must not trigger the discount.
        chunks = [{"search_type": "vector", "score": 0.1}]
        self.assertEqual(
            query_service.calculate_confidence(chunks),
            query_service.calculate_confidence(chunks, citation_count=None),
        )

    def test_distance_past_one_still_yields_meaningful_confidence(self):
        # Regression: a flat `min(distance, 1.0)` cap used to clamp any
        # distance >= 1.0 straight to 0% confidence, even though 1.0 is
        # not "unrelated" for a normalized-embedding L2 distance (the
        # real range is 0..2). A distance of 1.0098 - the exact value
        # observed for a correct, cited answer in manual testing - must
        # read as a real, non-zero score, not 0%.
        chunks = [{"search_type": "vector", "score": 1.0098}]
        self.assertEqual(query_service.calculate_confidence(chunks), 49)

    def test_zero_distance_yields_near_maximum_confidence(self):
        # An exact embedding match (distance 0) should read as ~100%,
        # clamped to the function's 99 ceiling.
        chunks = [{"search_type": "vector", "score": 0.0}]
        self.assertEqual(query_service.calculate_confidence(chunks), 99)

    def test_orthogonal_distance_yields_zero_confidence(self):
        # For unit-length embeddings, a distance of sqrt(2) corresponds
        # to a cosine similarity of exactly 0 (orthogonal / unrelated).
        chunks = [{"search_type": "vector", "score": 2 ** 0.5}]
        self.assertEqual(query_service.calculate_confidence(chunks), 0)

    def test_maximum_distance_clamps_to_zero_not_negative(self):
        # Distance 2.0 (exact opposite vectors) maps to cosine
        # similarity -1, which must clamp to 0%, not go negative.
        chunks = [{"search_type": "vector", "score": 2.0}]
        self.assertEqual(query_service.calculate_confidence(chunks), 0)


class QueryTransformServiceTests(unittest.TestCase):
    """
    generate_query_variants() / _parse_variants() - no live Gemini
    calls, offline and deterministic.
    """

    def test_parse_variants_includes_original_first_and_dedupes(self):
        raw = (
            '{"variants": ["What is the capital of France?", '
            '"what is the capital of france?", "Name the capital of France"]}'
        )

        variants = transform._parse_response(raw, "What is the capital of France?")

        self.assertEqual(variants[0], "What is the capital of France?")
        # Second entry is a case-only duplicate of the question and
        # must be dropped; only the genuinely new phrasing survives.
        self.assertEqual(variants, [
            "What is the capital of France?",
            "Name the capital of France",
        ])

    def test_parse_variants_malformed_json_falls_back_to_question(self):
        self.assertEqual(transform._parse_response("nope", "Q"), ["Q"])

    def test_generate_query_variants_skips_short_question(self):
        with patch.object(transform, "get_llm") as mock_get_llm:
            result = transform.generate_query_variants("hi")

        mock_get_llm.assert_not_called()
        self.assertEqual(result, ["hi"])

    def test_generate_query_variants_falls_back_on_llm_failure(self):
        with patch.object(transform, "get_llm", side_effect=RuntimeError("boom")):
            result = transform.generate_query_variants("What is the capital of France?")

        self.assertEqual(result, ["What is the capital of France?"])


class QueryExpansionServiceTests(unittest.TestCase):
    """
    expand_query() - no live Gemini calls (generate_query_variants is
    mocked directly).
    """

    def test_expand_query_adds_new_terms_from_variants(self):
        with patch.object(
            expansion,
            "generate_query_variants",
            return_value=["Who is the CEO?", "Who leads the company as chief executive?"],
        ):
            expanded = expansion.expand_query("Who is the CEO?")

        self.assertTrue(expanded.startswith("Who is the CEO?"))
        for term in ("leads", "company", "chief", "executive"):
            self.assertIn(term, expanded)

    def test_expand_query_returns_original_when_no_new_terms(self):
        with patch.object(
            expansion, "generate_query_variants", return_value=["Who is the CEO?"]
        ):
            self.assertEqual(expansion.expand_query("Who is the CEO?"), "Who is the CEO?")

    def test_expand_query_falls_back_on_failure(self):
        with patch.object(
            expansion, "generate_query_variants", side_effect=RuntimeError("boom")
        ):
            self.assertEqual(expansion.expand_query("Who is the CEO?"), "Who is the CEO?")

    def test_expand_query_empty_question(self):
        self.assertEqual(expansion.expand_query(""), "")


class HydeServiceTests(unittest.TestCase):
    """
    generate_hypothetical_document() - no live Gemini calls.
    """

    def test_generates_passage_from_mocked_response(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Paris is the capital of France."

        with patch.object(hyde, "get_llm", return_value=mock_llm):
            passage = hyde.generate_hypothetical_document("What is the capital of France?")

        self.assertEqual(passage, "Paris is the capital of France.")

    def test_skips_short_question(self):
        with patch.object(hyde, "get_llm") as mock_get_llm:
            result = hyde.generate_hypothetical_document("hi")

        mock_get_llm.assert_not_called()
        self.assertEqual(result, "")

    def test_returns_empty_string_on_failure(self):
        with patch.object(hyde, "get_llm", side_effect=RuntimeError("boom")):
            result = hyde.generate_hypothetical_document("What is the capital of France?")

        self.assertEqual(result, "")

    def test_truncates_overly_long_passage(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "x" * 5000

        with patch.object(hyde, "get_llm", return_value=mock_llm):
            passage = hyde.generate_hypothetical_document("What is the capital of France?")

        self.assertEqual(len(passage), hyde.MAX_HYPOTHETICAL_CHARS)


class DynamicTopKServiceTests(unittest.TestCase):
    """
    compute_dynamic_top_k() - pure heuristic, no DB/LLM.
    """

    def test_short_question_keeps_base_top_k(self):
        self.assertEqual(
            dynamic_topk.compute_dynamic_top_k("Who is the CEO?", base_top_k=3), 3
        )

    def test_long_question_increases_top_k(self):
        question = "What are the main findings of the Q3 report and how do they compare"
        self.assertGreater(
            dynamic_topk.compute_dynamic_top_k(question, base_top_k=3), 3
        )

    def test_multi_part_question_gets_extra_bump(self):
        single = dynamic_topk.compute_dynamic_top_k(
            "What is our revenue this quarter and its breakdown by region", base_top_k=3
        )
        multi_part = dynamic_topk.compute_dynamic_top_k(
            "What is our revenue this quarter and also what is next quarter's forecast?",
            base_top_k=3,
        )
        self.assertGreaterEqual(multi_part, single)

    def test_result_is_capped_at_max(self):
        from django.conf import settings as dj_settings

        question = "word " * 100 + "and also or plus as well as"
        result = dynamic_topk.compute_dynamic_top_k(question, base_top_k=8)
        self.assertLessEqual(result, dj_settings.DYNAMIC_TOP_K_MAX)

    def test_empty_question_returns_base(self):
        self.assertEqual(dynamic_topk.compute_dynamic_top_k("", base_top_k=4), 4)


class ReciprocalRankFusionTests(unittest.TestCase):
    """
    _reciprocal_rank_fusion() - pure ranking math, no DB/LLM.
    """

    def test_item_found_in_multiple_lists_ranks_first(self):
        list_a = [
            {"content": "a", "document": "Doc1", "chunk_number": 0, "score": 0.1, "search_type": "vector"},
            {"content": "b", "document": "Doc2", "chunk_number": 1, "score": 0.2, "search_type": "vector"},
        ]
        list_b = [
            {"content": "b", "document": "Doc2", "chunk_number": 1, "score": 5.0, "search_type": "bm25"},
            {"content": "c", "document": "Doc3", "chunk_number": 0, "score": 4.0, "search_type": "bm25"},
        ]

        fused = _reciprocal_rank_fusion([list_a, list_b], top_k=3)

        self.assertEqual((fused[0]["document"], fused[0]["chunk_number"]), ("Doc2", 1))
        self.assertTrue(all(item["search_type"] == "multi_query" for item in fused))

    def test_respects_top_k_cap(self):
        list_a = [
            {"content": str(i), "document": "Doc", "chunk_number": i, "score": 0.0, "search_type": "vector"}
            for i in range(5)
        ]
        fused = _reciprocal_rank_fusion([list_a], top_k=2)
        self.assertEqual(len(fused), 2)

    def test_empty_lists_produce_empty_result(self):
        self.assertEqual(_reciprocal_rank_fusion([], top_k=3), [])
        self.assertEqual(_reciprocal_rank_fusion([[], []], top_k=3), [])


class MultiQuerySearchTests(unittest.TestCase):
    """
    multi_query_search() orchestration - variant generation and the
    underlying vector_search()/bm25_search() calls are all mocked.
    """

    def test_returns_empty_when_no_extra_variants(self):
        with patch(
            "RAG.services.multi_query_service.generate_query_variants",
            return_value=["only the original"],
        ):
            self.assertEqual(multi_query_search("only the original"), [])

    def test_returns_empty_on_variant_generation_failure(self):
        with patch(
            "RAG.services.multi_query_service.generate_query_variants",
            side_effect=RuntimeError("boom"),
        ):
            self.assertEqual(multi_query_search("question"), [])

    def test_fuses_results_from_extra_variants(self):
        variant_result = [
            {"content": "x", "document": "Doc1", "chunk_number": 0, "score": 0.1, "search_type": "vector"}
        ]

        with patch(
            "RAG.services.multi_query_service.generate_query_variants",
            return_value=["question", "a variant"],
        ), patch(
            "RAG.services.multi_query_service.vector_search", return_value=variant_result
        ), patch(
            "RAG.services.multi_query_service.bm25_search", return_value=[]
        ):
            results = multi_query_search("question", top_k=3)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["search_type"], "multi_query")

    def test_one_failing_variant_does_not_abort_the_others(self):
        good_result = [
            {"content": "x", "document": "Doc1", "chunk_number": 0, "score": 0.1, "search_type": "vector"}
        ]

        call_count = {"n": 0}

        def flaky_vector_search(variant, top_k=None, filters=None, user=None, accessible_document_ids=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("transient failure")
            return good_result

        with patch(
            "RAG.services.multi_query_service.generate_query_variants",
            return_value=["question", "variant one", "variant two"],
        ), patch(
            "RAG.services.multi_query_service.vector_search", side_effect=flaky_vector_search
        ), patch(
            "RAG.services.multi_query_service.bm25_search", return_value=[]
        ):
            results = multi_query_search("question", top_k=3)

        self.assertEqual(len(results), 1)


class RetrievalFiltersTests(unittest.TestCase):
    """
    RetrievalFilters / apply_document_filters() - query construction
    only (never evaluated against the DB), so these run without a
    live database connection.
    """

    def test_from_request_parses_valid_document_id(self):
        filters = RetrievalFilters.from_request(document_ids=["5"])
        self.assertEqual(filters.document_ids, (5,))
        self.assertFalse(filters.is_empty())

    def test_from_request_parses_multiple_document_ids(self):
        filters = RetrievalFilters.from_request(document_ids=["5", "6"])
        self.assertEqual(filters.document_ids, (5, 6))
        self.assertFalse(filters.is_empty())

    def test_from_request_ignores_invalid_document_id(self):
        filters = RetrievalFilters.from_request(document_ids=["not-an-int"])
        self.assertIsNone(filters.document_ids)
        self.assertTrue(filters.is_empty())

    def test_from_request_blank_document_ids_is_no_filter(self):
        filters = RetrievalFilters.from_request(document_ids=[])
        self.assertTrue(filters.is_empty())

    def test_apply_document_filters_is_noop_for_none(self):
        queryset = DocumentChunk.objects.all()
        self.assertIs(apply_document_filters(queryset, None, document_field="document"), queryset)

    def test_apply_document_filters_is_noop_for_empty_filters(self):
        queryset = DocumentChunk.objects.all()
        empty_filters = RetrievalFilters()
        result = apply_document_filters(queryset, empty_filters, document_field="document")
        self.assertEqual(str(result.query), str(queryset.query))

    def test_apply_document_filters_builds_expected_lookup(self):
        filters = RetrievalFilters(document_ids=(5, 6))
        queryset = apply_document_filters(
            DocumentChunk.objects.all(), filters, document_field="document"
        )
        self.assertIn("document_id", str(queryset.query))
        self.assertIn("IN (5, 6)", str(queryset.query))

    def test_apply_document_filters_combines_file_type_and_date(self):
        filters = RetrievalFilters(
            file_types=("pdf",), uploaded_after=date(2026, 1, 1)
        )
        queryset = apply_document_filters(
            DocumentChunk.objects.all(), filters, document_field="document"
        )
        sql = str(queryset.query)
        self.assertIn("file_type", sql)
        self.assertIn("uploaded_at", sql)


class RerankerServiceTests(unittest.TestCase):
    """
    rerank_chunks() - the cross-encoder model itself is mocked, so
    these run offline and deterministically.
    """

    def setUp(self):
        self.chunks = [
            {"content": "irrelevant", "document": "D1", "chunk_number": 0, "score": 0.1, "search_type": "vector"},
            {"content": "very relevant", "document": "D1", "chunk_number": 1, "score": 0.5, "search_type": "bm25"},
            {"content": "somewhat relevant", "document": "D2", "chunk_number": 0, "score": 0.3, "search_type": "vector"},
        ]

    def test_reorders_by_descending_cross_encoder_score(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.1, 0.9, 0.5]

        with patch.object(reranker, "_get_reranker_model", return_value=mock_model):
            results = reranker.rerank_chunks("q", self.chunks)

        self.assertEqual(
            [r["content"] for r in results],
            ["very relevant", "somewhat relevant", "irrelevant"],
        )
        self.assertEqual(results[0]["rerank_score"], 0.9)

    def test_respects_top_k(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.1, 0.9, 0.5]

        with patch.object(reranker, "_get_reranker_model", return_value=mock_model):
            results = reranker.rerank_chunks("q", self.chunks, top_k=2)

        self.assertEqual(len(results), 2)

    def test_preserves_existing_keys(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.1, 0.9, 0.5]

        with patch.object(reranker, "_get_reranker_model", return_value=mock_model):
            results = reranker.rerank_chunks("q", self.chunks)

        self.assertEqual(results[0]["search_type"], "bm25")
        self.assertEqual(results[0]["document"], "D1")

    def test_does_not_mutate_input_chunks(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.1, 0.9, 0.5]

        with patch.object(reranker, "_get_reranker_model", return_value=mock_model):
            reranker.rerank_chunks("q", self.chunks)

        self.assertNotIn("rerank_score", self.chunks[0])

    def test_empty_input_passthrough(self):
        self.assertEqual(reranker.rerank_chunks("q", []), [])

    def test_falls_back_to_original_order_on_failure(self):
        with patch.object(
            reranker, "_get_reranker_model", side_effect=RuntimeError("model load failed")
        ):
            results = reranker.rerank_chunks("q", self.chunks)

        self.assertEqual(results, self.chunks)

    def test_falls_back_respects_top_k(self):
        with patch.object(reranker, "_get_reranker_model", side_effect=RuntimeError("boom")):
            results = reranker.rerank_chunks("q", self.chunks, top_k=1)

        self.assertEqual(results, self.chunks[:1])


class ContextCompressionServiceTests(unittest.TestCase):
    """
    compress_context() - generate_embedding() is mocked with plain
    vectors chosen so cosine similarity is easy to reason about, so
    these run offline and deterministically.
    """

    def setUp(self):
        self.chunks = [
            {"content": "the sky is blue", "document": "D1", "chunk_number": 0, "score": 0.1, "search_type": "vector"},
            {"content": "the sky is blue today", "document": "D1", "chunk_number": 1, "score": 0.2, "search_type": "bm25"},
            {"content": "the ocean is deep", "document": "D2", "chunk_number": 0, "score": 0.3, "search_type": "vector"},
        ]

    def test_drops_near_duplicate_keeps_first_occurrence(self):
        # Chunks 0 and 1 are identical in embedding space (redundant);
        # chunk 2 is orthogonal (distinct).
        embeddings = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]

        with patch.object(compression, "generate_embedding", side_effect=embeddings):
            results = compression.compress_context(self.chunks)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["content"], "the sky is blue")
        self.assertEqual(results[1]["content"], "the ocean is deep")

    def test_keeps_all_distinct_chunks(self):
        embeddings = [[1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]

        with patch.object(compression, "generate_embedding", side_effect=embeddings):
            results = compression.compress_context(self.chunks)

        self.assertEqual(len(results), 3)

    def test_single_chunk_returned_unchanged_without_embedding_call(self):
        with patch.object(compression, "generate_embedding") as m_embed:
            results = compression.compress_context(self.chunks[:1])

        self.assertEqual(results, self.chunks[:1])
        self.assertFalse(m_embed.called)

    def test_empty_list_returned_unchanged(self):
        self.assertEqual(compression.compress_context([]), [])

    def test_custom_threshold_overrides_settings(self):
        # Similarity 0.8 is below the default 0.92 threshold but above
        # a deliberately low custom threshold.
        embeddings = [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]]

        with patch.object(compression, "generate_embedding", side_effect=embeddings):
            results = compression.compress_context(self.chunks, similarity_threshold=0.5)

        self.assertEqual(len(results), 2)

    def test_falls_back_to_original_on_embedding_failure(self):
        with patch.object(compression, "generate_embedding", side_effect=RuntimeError("boom")):
            results = compression.compress_context(self.chunks)

        self.assertEqual(results, self.chunks)


class PromptTemplatesTests(unittest.TestCase):

    def test_is_not_found_answer_matches_fixed_fallback(self):
        self.assertTrue(
            prompt_templates.is_not_found_answer(prompt_templates.NOT_FOUND_ANSWER)
        )

    def test_is_not_found_answer_tolerant_of_whitespace_variation(self):
        self.assertTrue(
            prompt_templates.is_not_found_answer(
                "  I couldn't find the answer in the uploaded document.\n"
            )
        )

    def test_is_not_found_answer_false_for_real_answer(self):
        self.assertFalse(
            prompt_templates.is_not_found_answer("The launch date is March 2024 [1].")
        )

    def test_is_not_found_answer_false_for_empty_or_none(self):
        self.assertFalse(prompt_templates.is_not_found_answer(""))
        self.assertFalse(prompt_templates.is_not_found_answer(None))

    def test_build_answer_prompt_embeds_context_question_and_fallback(self):
        prompt = prompt_templates.build_answer_prompt(
            "[1] (Doc, chunk 0):\nSome content", "What is it?"
        )

        self.assertIn("[1] (Doc, chunk 0):\nSome content", prompt)
        self.assertIn("What is it?", prompt)
        self.assertIn(prompt_templates.NOT_FOUND_ANSWER, prompt)


class CitationServiceTests(unittest.TestCase):

    def setUp(self):
        self.chunks = [
            {"content": "Revenue grew 20%.", "document": "Q3 Report", "chunk_number": 0, "score": 0.1, "search_type": "vector"},
            {"content": "Costs fell 5%.", "document": "Q3 Report", "chunk_number": 1, "score": 0.2, "search_type": "bm25"},
        ]

    def test_build_cited_context_numbers_chunks_in_order(self):
        context = citation_service.build_cited_context(self.chunks)

        self.assertIn("[1] (Q3 Report, chunk 0):\nRevenue grew 20%.", context)
        self.assertIn("[2] (Q3 Report, chunk 1):\nCosts fell 5%.", context)
        self.assertLess(context.index("[1]"), context.index("[2]"))

    def test_extract_citations_returns_cited_chunks_in_order(self):
        answer = "Costs fell [2], and revenue grew [1]."

        citations = citation_service.extract_citations(answer, self.chunks)

        self.assertEqual([c["chunk_number"] for c in citations], [0, 1])
        self.assertEqual(citations[0]["citation_number"], 1)
        self.assertEqual(citations[1]["citation_number"], 2)

    def test_extract_citations_dedups_repeated_markers(self):
        answer = "Revenue grew [1]. It really grew [1]!"

        citations = citation_service.extract_citations(answer, self.chunks)

        self.assertEqual(len(citations), 1)

    def test_extract_citations_drops_out_of_range_numbers(self):
        answer = "Revenue grew [1], allegedly per [99]."

        citations = citation_service.extract_citations(answer, self.chunks)

        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0]["citation_number"], 1)

    def test_extract_citations_empty_for_uncited_answer(self):
        citations = citation_service.extract_citations("No markers here.", self.chunks)
        self.assertEqual(citations, [])

    def test_extract_citations_empty_for_no_answer_or_chunks(self):
        self.assertEqual(citation_service.extract_citations("", self.chunks), [])
        self.assertEqual(citation_service.extract_citations("[1]", []), [])


class LlmServiceTests(unittest.TestCase):
    """
    generate_answer() now returns a (answer, extras) tuple from a
    structured JSON-mode response (see llm_service.py's docstring),
    routed through the multi-provider llm_client.get_llm().generate()
    rather than a single-provider get_model()/generate_content() call.
    """

    def test_generate_answer_returns_stripped_model_text(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = '{"answer": "  The answer is 42 [1].  "}'

        with patch.object(llm_service, "get_llm", return_value=mock_llm):
            answer, extras = llm_service.generate_answer("[1] (Doc, chunk 0):\ncontext", "Q?")

        self.assertEqual(answer, "The answer is 42 [1].")

    def test_generate_answer_passes_configured_temperature(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = '{"answer": "answer"}'

        with patch.object(llm_service, "get_llm", return_value=mock_llm), \
             override_settings(ANSWER_TEMPERATURE=0.1):

            llm_service.generate_answer("context", "Q?")

        _, kwargs = mock_llm.generate.call_args
        self.assertEqual(kwargs["temperature"], 0.1)

    def test_generate_answer_returns_not_found_on_empty_response(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = ""

        with patch.object(llm_service, "get_llm", return_value=mock_llm):
            answer, extras = llm_service.generate_answer("context", "Q?")

        self.assertEqual(answer, prompt_templates.NOT_FOUND_ANSWER)

    def test_generate_answer_returns_service_unavailable_when_every_provider_fails(self):
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = llm_service.AllProvidersFailedError("boom")

        with patch.object(llm_service, "get_llm", return_value=mock_llm):
            answer, extras = llm_service.generate_answer("context", "Q?")

        self.assertEqual(answer, prompt_templates.SERVICE_UNAVAILABLE_ANSWER)

    def test_generate_answer_returns_not_found_on_unexpected_failure(self):
        with patch.object(llm_service, "get_llm", side_effect=RuntimeError("boom")):
            answer, extras = llm_service.generate_answer("context", "Q?")

        self.assertEqual(answer, prompt_templates.NOT_FOUND_ANSWER)


class RetrieveChunksOrchestrationTests(unittest.TestCase):
    """
    retrieve_chunks() feature-flag orchestration. All underlying
    retrieval sources (vector/bm25/graph/hyde/multi_query) are
    mocked, so this validates the wiring - which sources get called,
    with what arguments, and how results merge - without touching the
    database.
    """

    def setUp(self):
        # retrieve_chunks() caches its result per (question, user,
        # filters, top_k) - see retrieval_service.py -
        # process-wide/module-level, not per-test-isolated the way the
        # DB is for a django.test.TestCase. Several tests below reuse
        # the exact same "Who is the CEO?" question with no explicit
        # top_k, so without clearing here, whichever test happens to
        # run first "wins" and every later one silently gets its
        # mocked result back instead of exercising its own mocks.
        cache.clear()

        self.vector_result = [{"content": "v", "document": "D1", "chunk_number": 0, "score": 0.1, "search_type": "vector"}]
        self.bm25_result = [{"content": "b", "document": "D1", "chunk_number": 1, "score": 2.0, "search_type": "bm25"}]
        self.hyde_result = [{"content": "h", "document": "D2", "chunk_number": 0, "score": 0.2, "search_type": "hyde"}]
        self.mq_result = [{"content": "m", "document": "D3", "chunk_number": 0, "score": 0.03, "search_type": "multi_query"}]

    def test_default_flags_skip_hyde_and_multi_query(self):
        with patch.object(retrieval, "vector_search", return_value=self.vector_result), \
             patch.object(retrieval, "bm25_search", return_value=self.bm25_result) as m_bm25, \
             patch.object(retrieval, "graph_search", return_value=[]), \
             patch.object(retrieval, "hyde_search", return_value=self.hyde_result) as m_hyde, \
             patch("RAG.services.multi_query_service.multi_query_search", return_value=self.mq_result) as m_mq:

            results = retrieval.retrieve_chunks("Who is the CEO?")

        self.assertFalse(m_hyde.called)
        self.assertFalse(m_mq.called)
        self.assertEqual({r["search_type"] for r in results}, {"vector", "bm25"})
        # BM25 receives the raw question unchanged when expansion is off.
        self.assertEqual(m_bm25.call_args.args[0], "Who is the CEO?")

    @override_settings(ENABLE_HYDE=True, ENABLE_MULTI_QUERY=True, ENABLE_QUERY_EXPANSION=True)
    def test_enabling_all_flags_wires_every_source(self):
        with patch.object(retrieval, "vector_search", return_value=self.vector_result), \
             patch.object(retrieval, "bm25_search", return_value=self.bm25_result) as m_bm25, \
             patch.object(retrieval, "graph_search", return_value=[]), \
             patch.object(retrieval, "hyde_search", return_value=self.hyde_result) as m_hyde, \
             patch.object(retrieval, "expand_query", return_value="who is the ceo boss") as m_expand, \
             patch("RAG.services.multi_query_service.multi_query_search", return_value=self.mq_result) as m_mq:

            results = retrieval.retrieve_chunks("Who is the CEO?", top_k=10)

        self.assertTrue(m_hyde.called)
        self.assertTrue(m_mq.called)
        self.assertTrue(m_expand.called)
        self.assertEqual(
            {r["search_type"] for r in results},
            {"vector", "bm25", "hyde", "multi_query"},
        )
        # BM25 receives the expanded query text, not the raw question.
        self.assertEqual(m_bm25.call_args.args[0], "who is the ceo boss")

    def test_explicit_top_k_overrides_dynamic_sizing(self):
        with patch.object(retrieval, "vector_search", return_value=[]) as m_vec, \
             patch.object(retrieval, "bm25_search", return_value=[]) as m_bm25, \
             patch.object(retrieval, "graph_search", return_value=[]):

            retrieval.retrieve_chunks("Who is the CEO?", top_k=7)

        self.assertEqual(m_vec.call_args.kwargs["top_k"], 7)
        self.assertEqual(m_bm25.call_args.args[1], 7)

    @override_settings(ENABLE_DYNAMIC_TOP_K=False)
    def test_dynamic_top_k_disabled_uses_fixed_settings_top_k(self):
        from django.conf import settings as dj_settings

        with patch.object(retrieval, "vector_search", return_value=[]) as m_vec, \
             patch.object(retrieval, "bm25_search", return_value=[]), \
             patch.object(retrieval, "graph_search", return_value=[]):

            retrieval.retrieve_chunks(
                "What are the findings and risks and also the mitigations discussed?"
            )

        self.assertEqual(m_vec.call_args.kwargs["top_k"], dj_settings.TOP_K)

    def test_reranker_disabled_by_default_skips_rerank(self):
        with patch.object(retrieval, "vector_search", return_value=self.vector_result), \
             patch.object(retrieval, "bm25_search", return_value=self.bm25_result), \
             patch.object(retrieval, "graph_search", return_value=[]), \
             patch.object(retrieval, "rerank_chunks") as m_rerank:

            retrieval.retrieve_chunks("Who is the CEO?")

        self.assertFalse(m_rerank.called)

    @override_settings(ENABLE_RERANKER=True, RERANKER_CANDIDATE_MULTIPLIER=4)
    def test_reranker_enabled_over_fetches_and_delegates_final_result(self):
        with patch.object(retrieval, "vector_search", return_value=self.vector_result) as m_vec, \
             patch.object(retrieval, "bm25_search", return_value=self.bm25_result) as m_bm25, \
             patch.object(retrieval, "graph_search", return_value=[]) as m_graph, \
             patch.object(retrieval, "rerank_chunks", return_value=["reranked"]) as m_rerank:

            results = retrieval.retrieve_chunks("Who is the CEO?", top_k=3)

        # Candidate pool is over-fetched by RERANKER_CANDIDATE_MULTIPLIER
        # (3 * 4 = 12) so the reranker has real alternatives to reorder.
        self.assertEqual(m_vec.call_args.kwargs["top_k"], 12)
        self.assertEqual(m_bm25.call_args.args[1], 12)
        self.assertEqual(m_graph.call_args.args[2], 12)

        self.assertTrue(m_rerank.called)
        self.assertEqual(m_rerank.call_args.args[0], "Who is the CEO?")
        self.assertEqual(m_rerank.call_args.kwargs["top_k"], 3)
        self.assertEqual(results, ["reranked"])


class AnswerQuestionCompressionTests(unittest.TestCase):
    """
    answer_question()'s context-compression integration.
    retrieve_chunks() and generate_answer() are mocked, and `user`
    stays None so QueryLog.objects.create() is never reached, so this
    runs fully offline without a database.
    """

    def setUp(self):
        self.chunks = [
            {"content": "chunk a", "document": "D1", "chunk_number": 0, "score": 0.1, "search_type": "vector"},
            {"content": "chunk b", "document": "D1", "chunk_number": 1, "score": 0.2, "search_type": "vector"},
        ]

    def test_compression_disabled_by_default_skips_compress_context(self):
        with patch.object(query_service, "retrieve_chunks", return_value=self.chunks), \
             patch.object(query_service, "generate_answer", return_value=("the answer", {})) as m_answer, \
             patch.object(query_service, "compress_context") as m_compress:

            result = query_service.answer_question("Who is the CEO?")

        self.assertFalse(m_compress.called)
        self.assertEqual(result["sources"], self.chunks)
        context_arg = m_answer.call_args.args[0]
        self.assertIn("chunk a", context_arg)
        self.assertIn("chunk b", context_arg)

    @override_settings(ENABLE_CONTEXT_COMPRESSION=True)
    def test_compression_enabled_filters_before_context_is_built(self):
        compressed = [self.chunks[0]]

        with patch.object(query_service, "retrieve_chunks", return_value=self.chunks), \
             patch.object(query_service, "generate_answer", return_value=("the answer", {})) as m_answer, \
             patch.object(query_service, "compress_context", return_value=compressed) as m_compress:

            result = query_service.answer_question("Who is the CEO?")

        self.assertTrue(m_compress.called)
        self.assertEqual(m_compress.call_args.args[0], self.chunks)
        self.assertEqual(result["sources"], compressed)
        context_arg = m_answer.call_args.args[0]
        self.assertIn("chunk a", context_arg)
        self.assertNotIn("chunk b", context_arg)


class AnswerQuestionCitationTests(unittest.TestCase):
    """
    answer_question()'s Sprint 9 citation + confidence integration.
    retrieve_chunks() and generate_answer() are mocked, and `user`
    stays None so QueryLog.objects.create() is never reached, so this
    runs fully offline without a database.
    """

    def setUp(self):
        self.chunks = [
            {"content": "Revenue grew 20%.", "document": "Q3 Report", "chunk_number": 0, "score": 0.05, "search_type": "vector"},
            {"content": "Costs fell 5%.", "document": "Q3 Report", "chunk_number": 1, "score": 0.4, "search_type": "vector"},
        ]

    def test_cited_answer_populates_citations_and_marks_sources(self):
        with patch.object(query_service, "retrieve_chunks", return_value=self.chunks), \
             patch.object(query_service, "generate_answer", return_value=("Revenue grew 20% [1].", {})):

            result = query_service.answer_question("How did revenue change?")

        self.assertEqual(len(result["citations"]), 1)
        self.assertEqual(result["citations"][0]["chunk_number"], 0)
        self.assertEqual(result["sources"][0]["citation_number"], 1)
        self.assertNotIn("citation_number", result["sources"][1])

    def test_uncited_answer_discounts_confidence_relative_to_cited(self):
        with patch.object(query_service, "retrieve_chunks", return_value=self.chunks), \
             patch.object(query_service, "generate_answer", return_value=("Revenue grew 20% [1].", {})):

            cited_result = query_service.answer_question("How did revenue change?")

        with patch.object(query_service, "retrieve_chunks", return_value=self.chunks), \
             patch.object(query_service, "generate_answer", return_value=("Revenue grew, apparently.", {})):

            uncited_result = query_service.answer_question("How did revenue change?")

        self.assertEqual(uncited_result["citations"], [])
        self.assertLess(uncited_result["confidence"], cited_result["confidence"])

    def test_not_found_answer_yields_zero_confidence_and_no_citations(self):
        with patch.object(query_service, "retrieve_chunks", return_value=self.chunks), \
             patch.object(
                 query_service, "generate_answer", return_value=(prompt_templates.NOT_FOUND_ANSWER, {})
             ):

            result = query_service.answer_question("What is the capital of Mars?")

        self.assertEqual(result["confidence"], 0)
        self.assertEqual(result["citations"], [])


class HealthServiceTests(unittest.TestCase):
    """
    get_health_status() - get_system_status(), the background task pool
    check, and the LLM provider health checks are all mocked, so this
    runs offline regardless of whether a real database/LLM provider is
    reachable.
    """

    def setUp(self):
        # Every test in this class exercises DB/background-pool logic,
        # not LLM provider logic - default to "nothing configured" (an
        # empty dict short-circuits get_health_status()'s `if
        # llm_providers:` check) so these stay offline and each test's
        # pre-existing assertions are unaffected. See
        # LlmProviderHealthCheckTests below for coverage of this
        # check's own behavior.
        patcher = patch.object(health_service, "_check_llm_providers", return_value={})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _system_status(self, db_online=True, pgvector_enabled=True, embeddings_complete=True):
        return {
            "db_online": db_online,
            "pgvector_enabled": pgvector_enabled,
            "embeddings_complete": embeddings_complete,
        }

    def _bg_jobs(self, available=True):
        return {"available": available, "max_workers": 4, "active": 0, "pending": 0}

    def test_ok_when_db_and_pgvector_and_background_pool_healthy(self):
        with patch.object(health_service, "get_system_status", return_value=self._system_status()), \
             patch.object(health_service, "_check_background_jobs", return_value=self._bg_jobs()):

            result = health_service.get_health_status()

        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["checks"]["database"])
        self.assertTrue(result["checks"]["background_jobs"])

    def test_degraded_when_pgvector_disabled(self):
        with patch.object(
            health_service, "get_system_status",
            return_value=self._system_status(pgvector_enabled=False),
        ), patch.object(health_service, "_check_background_jobs", return_value=self._bg_jobs()):

            result = health_service.get_health_status()

        self.assertEqual(result["status"], "degraded")

    def test_degraded_when_background_pool_unavailable(self):
        with patch.object(health_service, "get_system_status", return_value=self._system_status()), \
             patch.object(health_service, "_check_background_jobs", return_value=self._bg_jobs(available=False)):

            result = health_service.get_health_status()

        self.assertEqual(result["status"], "degraded")
        self.assertFalse(result["checks"]["background_jobs"])

    def test_never_raises_when_system_status_blows_up(self):
        with patch.object(
            health_service, "get_system_status", side_effect=RuntimeError("db exploded")
        ), patch.object(health_service, "_check_background_jobs", return_value=self._bg_jobs()):

            result = health_service.get_health_status()

        self.assertEqual(result["status"], "degraded")
        self.assertFalse(result["checks"]["database"])


class LlmProviderHealthCheckTests(unittest.TestCase):
    """
    _check_llm_providers() and its fold-in to get_health_status()'s
    overall verdict - added alongside the performance/infra audit's
    ask_ai_stream/check_infra work ("all configured LLM providers
    before the application is considered healthy"). get_llm() and
    _is_configured() are both mocked, so this never makes a real
    provider network call.
    """

    def _system_status(self):
        return {"db_online": True, "pgvector_enabled": True, "embeddings_complete": True}

    def _bg_jobs(self):
        return {"available": True, "max_workers": 4, "active": 0, "pending": 0}

    def test_no_providers_configured_returns_empty_dict(self):
        with patch.object(health_service, "_is_configured", return_value=False):
            self.assertEqual(health_service._check_llm_providers(), {})

    def test_checks_only_configured_providers(self):
        mock_llm = MagicMock()
        mock_llm.health_check.side_effect = lambda provider: {"ok": provider == "openrouter", "latency_ms": 42, "message": "Connected"}

        with patch.object(health_service, "_is_configured", side_effect=lambda p: p in ("openrouter", "gemini")), \
             patch.object(health_service, "get_llm", return_value=mock_llm):

            result = health_service._check_llm_providers()

        self.assertEqual(result["openrouter"]["ok"], True)
        self.assertEqual(result["gemini"]["ok"], False)
        self.assertNotIn("groq", result)

    def test_provider_check_exception_reports_false_not_raise(self):
        mock_llm = MagicMock()
        mock_llm.health_check.side_effect = RuntimeError("boom")

        with patch.object(health_service, "_is_configured", return_value=True), \
             patch.object(health_service, "get_llm", return_value=mock_llm):

            result = health_service._check_llm_providers()

        self.assertTrue(result)
        self.assertTrue(all(entry["ok"] is False for entry in result.values()))

    def test_no_configured_providers_does_not_block_overall_health(self):
        with patch.object(health_service, "get_system_status", return_value=self._system_status()), \
             patch.object(health_service, "_check_background_jobs", return_value=self._bg_jobs()), \
             patch.object(health_service, "_check_llm_providers", return_value={}):

            result = health_service.get_health_status()

        self.assertEqual(result["status"], "ok")

    def test_at_least_one_healthy_provider_required_when_any_configured(self):
        # live_llm_check=True exercises _check_llm_providers() (mocked
        # below) - the default False path calls
        # _recent_llm_provider_status() instead (usage-derived, see
        # its own tests), but the "at least one must be healthy"
        # aggregation logic in get_health_status() is shared by both,
        # so testing it through either call is equally valid.
        with patch.object(health_service, "get_system_status", return_value=self._system_status()), \
             patch.object(health_service, "_check_background_jobs", return_value=self._bg_jobs()), \
             patch.object(health_service, "_check_llm_providers", return_value={
                 "openrouter": {"ok": False, "latency_ms": None, "message": "down"},
                 "gemini": {"ok": False, "latency_ms": None, "message": "down"},
             }):

            result = health_service.get_health_status(live_llm_check=True)

        self.assertEqual(result["status"], "degraded")

    def test_one_healthy_provider_among_several_keeps_overall_status_ok(self):
        healthy_providers = {
            "openrouter": {"ok": True, "latency_ms": 120, "message": "Connected"},
            "gemini": {"ok": False, "latency_ms": None, "message": "down"},
        }

        with patch.object(health_service, "get_system_status", return_value=self._system_status()), \
             patch.object(health_service, "_check_background_jobs", return_value=self._bg_jobs()), \
             patch.object(health_service, "_check_llm_providers", return_value=healthy_providers):

            result = health_service.get_health_status(live_llm_check=True)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["checks"]["llm_providers"], healthy_providers)

    def test_recent_provider_status_null_ok_does_not_block_overall_health(self):
        """A configured provider with zero recent traffic (ok=None) must not, by itself, drag the overall verdict to "degraded" - "no data" isn't "bad data". This is the default (live_llm_check=False) path."""

        with patch.object(health_service, "get_system_status", return_value=self._system_status()), \
             patch.object(health_service, "_check_background_jobs", return_value=self._bg_jobs()), \
             patch.object(health_service, "_recent_llm_provider_status", return_value={
                 "openrouter": {"ok": None, "latency_ms": None, "message": "No requests in the last 15 minutes - use Check Now for a live check."},
             }):

            result = health_service.get_health_status()

        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["live_llm_check"])


class LLMClientFallbackChainTests(unittest.TestCase):
    """
    LLMClient._build_chain() - pure function of settings + which
    provider API keys are configured, no network/DB. Locks in the
    guarantee _build_chain()'s own docstring promises ("never silently
    substitutes a different provider the admin didn't select"): with
    fallback disabled, a broken/unconfigured primary must fail rather
    than quietly trying Groq/Gemini/whichever other provider happens
    to have a key configured. This is the actual mechanism behind "AI
    Tasks must use the selected model, not silently use Gemini until
    it's explicitly selected" - the toggle already exists and already
    works; these tests are what verify that going forward.
    """

    @override_settings(LLM_PROVIDER="openrouter", LLM_FALLBACK_ENABLED=False,
                        OPENROUTER_API_KEY="key", GEMINI_API_KEY="key", GROQ_API_KEY="")
    def test_fallback_disabled_returns_only_the_primary(self):
        chain = llm_client.LLMClient()._build_chain()
        self.assertEqual(chain, ["openrouter"])

    @override_settings(LLM_PROVIDER="openrouter", LLM_FALLBACK_ENABLED=False,
                        OPENROUTER_API_KEY="", GEMINI_API_KEY="key", GROQ_API_KEY="key")
    def test_fallback_disabled_and_primary_unconfigured_returns_empty_chain(self):
        # Even though Gemini and Groq both have keys configured, a
        # disabled fallback must never substitute either of them for
        # an unconfigured primary - an empty chain (which
        # LLMClient.generate() turns into a clear AllProvidersFailedError)
        # is the correct outcome, not a silent switch to Gemini.
        chain = llm_client.LLMClient()._build_chain()
        self.assertEqual(chain, [])

    @override_settings(LLM_PROVIDER="openrouter", LLM_FALLBACK_ENABLED=True,
                        OPENROUTER_API_KEY="key", GEMINI_API_KEY="key", GROQ_API_KEY="")
    def test_fallback_enabled_appends_remaining_configured_providers_after_primary(self):
        chain = llm_client.LLMClient()._build_chain()
        self.assertEqual(chain, ["openrouter", "gemini"])

    @override_settings(LLM_PROVIDER="gemini", LLM_FALLBACK_ENABLED=True,
                        OPENROUTER_API_KEY="key", GEMINI_API_KEY="key", GROQ_API_KEY="key")
    def test_primary_is_not_duplicated_in_the_fallback_tail(self):
        chain = llm_client.LLMClient()._build_chain()
        self.assertEqual(chain, ["gemini", "groq", "openrouter"])

    @override_settings(LLM_PROVIDER="not_a_real_provider", LLM_FALLBACK_ENABLED=False,
                        OPENROUTER_API_KEY="key", GEMINI_API_KEY="key", GROQ_API_KEY="key")
    def test_unknown_primary_provider_is_ignored_not_substituted(self):
        chain = llm_client.LLMClient()._build_chain()
        self.assertEqual(chain, [])


class ProcessDocumentTaskTests(unittest.TestCase):
    """
    RAG.tasks.process_document_task - Document.objects.get() and
    process_uploaded_document() are mocked. Calling the task directly
    (not via task_runner.submit()) runs it synchronously with no thread
    pool required.
    """

    def test_processes_existing_document(self):
        mock_document = MagicMock(id=7)

        with patch.object(tasks.Document.objects, "get", return_value=mock_document) as m_get, \
             patch.object(tasks, "process_uploaded_document") as m_process:

            tasks.process_document_task(7)

        m_get.assert_called_once_with(id=7)
        m_process.assert_called_once_with(mock_document)

    def test_logs_and_returns_when_document_missing(self):
        with patch.object(
            tasks.Document.objects, "get", side_effect=tasks.Document.DoesNotExist
        ), patch.object(tasks, "process_uploaded_document") as m_process:

            result = tasks.process_document_task(999)

        self.assertIsNone(result)
        self.assertFalse(m_process.called)

    def test_retries_then_gives_up_without_raising(self):
        mock_document = MagicMock(id=7)

        with patch.object(tasks.Document.objects, "get", return_value=mock_document), \
             patch.object(
                 tasks, "process_uploaded_document", side_effect=RuntimeError("boom")
             ) as m_process, \
             patch.object(tasks.time, "sleep"):

            result = tasks.process_document_task(7)

        self.assertIsNone(result)
        self.assertEqual(m_process.call_count, tasks.MAX_PROCESSING_RETRIES)


class MaskEmailTests(unittest.TestCase):
    """RAG.utils.formatting.mask_email() - pure string logic, no DB."""

    def test_masks_middle_of_local_part(self):
        from .utils.formatting import mask_email
        self.assertEqual(mask_email("johndoe@example.com"), "j*****e@example.com")

    def test_short_local_part(self):
        from .utils.formatting import mask_email
        self.assertEqual(mask_email("ab@example.com"), "a*@example.com")

    def test_single_char_local_part(self):
        from .utils.formatting import mask_email
        self.assertEqual(mask_email("a@example.com"), "a*@example.com")

    def test_empty_or_invalid_input(self):
        from .utils.formatting import mask_email
        self.assertEqual(mask_email(""), "")
        self.assertEqual(mask_email(None), "")
        self.assertEqual(mask_email("not-an-email"), "not-an-email")


class OtpCodeHashRoundTripTests(unittest.TestCase):
    """
    otp_service generates a code and stores only make_password(code) -
    confirms check_password() round-trips correctly and a wrong code
    never matches. No DB needed, just Django's password hasher.
    """

    def test_generated_code_is_six_digits(self):
        from .services.otp_service import _generate_code, OTP_LENGTH
        code = _generate_code()
        self.assertEqual(len(code), OTP_LENGTH)
        self.assertTrue(code.isdigit())

    def test_hash_round_trip(self):
        from django.contrib.auth.hashers import check_password, make_password
        from .services.otp_service import _generate_code

        code = _generate_code()
        hashed = make_password(code)

        self.assertNotEqual(hashed, code)  # never stored in plaintext
        self.assertTrue(check_password(code, hashed))
        self.assertFalse(check_password("000000" if code != "000000" else "111111", hashed))


class RateLimitServiceTests(unittest.TestCase):
    """Fixed-window counter logic against the real (LocMemCache-backed) Django cache - no DB needed."""

    def setUp(self):
        cache.clear()

    def test_allows_up_to_limit_then_blocks(self):
        from .services.rate_limit_service import is_rate_limited

        key = "test:allows_up_to_limit"
        for _ in range(3):
            self.assertFalse(is_rate_limited(key, limit=3, window_seconds=60))
        self.assertTrue(is_rate_limited(key, limit=3, window_seconds=60))

    def test_independent_keys_dont_interfere(self):
        from .services.rate_limit_service import is_rate_limited

        for _ in range(3):
            is_rate_limited("test:key_a", limit=3, window_seconds=60)

        self.assertFalse(is_rate_limited("test:key_b", limit=3, window_seconds=60))

    def test_cooldown_starts_and_reports_remaining(self):
        from .services.rate_limit_service import get_cooldown_remaining_seconds, start_cooldown

        key = "test:cooldown"
        self.assertEqual(get_cooldown_remaining_seconds(key), 0)
        start_cooldown(key, 60)
        remaining = get_cooldown_remaining_seconds(key)
        self.assertTrue(0 < remaining <= 60)


class NotificationServiceTests(TestCase):
    """create_notification()/mark_read()/mark_all_read()/get_unread_count() against a real test DB."""

    def setUp(self):
        cache.clear()
        self.recipient = User.objects.create_user(username="notif_recipient", password="pw", email="recipient@example.com")
        self.actor = User.objects.create_user(username="notif_actor", password="pw")

    def test_create_notification_creates_row(self):
        from .services.notification_service import create_notification

        notification = create_notification(
            recipient=self.recipient, actor=self.actor, notification_type="document.shared",
            title="Test", message="Test message", send_email=False,
        )
        self.assertIsNotNone(notification)
        self.assertEqual(Notification.objects.filter(recipient=self.recipient).count(), 1)
        self.assertFalse(notification.is_read)

    def test_create_notification_never_raises_for_invalid_recipient(self):
        from .services.notification_service import create_notification

        result = create_notification(
            recipient=None, notification_type="document.shared", title="T", message="M",
        )
        self.assertIsNone(result)

    def test_mark_read_is_ownership_scoped(self):
        from .services.notification_service import mark_read

        notification = Notification.objects.create(
            recipient=self.recipient, notification_type="document.shared", title="T", message="M",
        )
        other_user = User.objects.create_user(username="notif_other", password="pw")

        self.assertFalse(mark_read(notification.id, other_user))
        notification.refresh_from_db()
        self.assertFalse(notification.is_read)

        self.assertTrue(mark_read(notification.id, self.recipient))
        notification.refresh_from_db()
        self.assertTrue(notification.is_read)

    def test_mark_all_read_and_unread_count(self):
        from .services.notification_service import get_unread_count, mark_all_read

        for i in range(3):
            Notification.objects.create(recipient=self.recipient, notification_type="document.shared", title=f"T{i}", message="M")

        self.assertEqual(get_unread_count(self.recipient), 3)
        marked = mark_all_read(self.recipient)
        self.assertEqual(marked, 3)
        self.assertEqual(get_unread_count(self.recipient), 0)


class DocumentShareConstraintTests(TestCase):
    """
    DocumentShare's 3-way exactly-one-target CheckConstraint and the
    partial (invited_email-only) UniqueConstraint added for invite-by-
    email sharing (Phase 7) - regression coverage for the bug where a
    plain unique_together on invited_email collided across every
    ordinary (blank-invited_email) share on the same document.
    """

    def setUp(self):
        self.owner = User.objects.create_user(username="share_owner", password="pw")
        self.other = User.objects.create_user(username="share_other", password="pw")
        self.document = Document.objects.create(user=self.owner, title="Doc", file="documents/test.txt")

    def test_two_ordinary_shares_on_same_document_do_not_collide(self):
        """Regression: both blank invited_email - must NOT trip the partial unique constraint."""

        DocumentShare.objects.create(document=self.document, shared_with_user=self.other, shared_by=self.owner)

        third = User.objects.create_user(username="share_third", password="pw")
        role_share = DocumentShare.objects.create(document=self.document, invited_email="pending@example.com", shared_by=self.owner)

        self.assertEqual(DocumentShare.objects.filter(document=self.document).count(), 2)

    def test_duplicate_pending_invite_rejected_at_db_level(self):
        DocumentShare.objects.create(document=self.document, invited_email="dup@example.com", shared_by=self.owner)

        with self.assertRaises(Exception):
            DocumentShare.objects.create(document=self.document, invited_email="dup@example.com", shared_by=self.owner)

    def test_create_share_email_branch_creates_pending_invite(self):
        from .services.sharing_service import create_share

        share = create_share(self.document, self.owner, "email", "invitee@example.com")
        self.assertEqual(share.invited_email, "invitee@example.com")
        self.assertIsNone(share.shared_with_user)

    def test_create_share_email_branch_resolves_to_existing_user(self):
        from .services.sharing_service import create_share

        existing = User.objects.create_user(username="already_here", password="pw", email="already@example.com")
        share = create_share(self.document, self.owner, "email", "already@example.com")
        self.assertEqual(share.shared_with_user_id, existing.id)
        self.assertEqual(share.invited_email, "")


class OtpInviteConversionTests(TestCase):
    """otp_service.verify_otp() converting a pending DocumentShare.invited_email into a real share on successful verification (Phase 7)."""

    def setUp(self):
        cache.clear()
        self.owner = User.objects.create_user(username="convert_owner", password="pw")
        self.document = Document.objects.create(user=self.owner, title="Doc", file="documents/test.txt")
        self.invitee = User.objects.create_user(
            username="convert_invitee", password="pw", email="convertme@example.com", is_active=False,
        )
        self.share = DocumentShare.objects.create(
            document=self.document, invited_email="convertme@example.com", shared_by=self.owner,
        )

    def test_verify_otp_converts_pending_invite_and_notifies(self):
        from .services import otp_service

        otp_service.generate_and_send_otp(self.invitee)
        otp = self.invitee.email_otps.filter(is_used=False).latest("created_at")

        # Recover the raw code the same way the real flow would never
        # need to (it only ever exists in-memory/in the email) - here
        # we bypass by generating our own OTP row directly instead of
        # trying to intercept the background-emailed code.
        from django.contrib.auth.hashers import make_password
        raw_code = "123456"
        otp.code_hash = make_password(raw_code)
        otp.save(update_fields=["code_hash"])

        success, status = otp_service.verify_otp(self.invitee, raw_code)

        self.assertTrue(success)
        self.assertEqual(status, "")

        self.share.refresh_from_db()
        self.assertEqual(self.share.shared_with_user_id, self.invitee.id)
        self.assertEqual(self.share.invited_email, "")

        self.assertTrue(
            Notification.objects.filter(recipient=self.invitee, notification_type="document.shared").exists()
        )


class ExecuteRunStatusTests(TestCase):
    """
    ai_tasks_engine_service.execute_run()'s COMPLETED vs FAILED
    decision. Added as a regression test: a run where every single
    item failed used to still be marked COMPLETED (with only a
    best-effort note in error_message), so a run that produced zero
    real results - e.g. every configured LLM provider down or
    misconfigured - still surfaced as "AI Task completed" instead of
    "AI Task failed". get_document_context_text() and _call_llm_json()
    are both mocked so this never touches the filesystem or a real LLM
    provider.
    """

    def setUp(self):
        self.user = User.objects.create_user(username="ai_tasks_tester", password="pw")
        self.doc_a = Document.objects.create(user=self.user, title="Doc A", file="documents/a.txt")
        self.doc_b = Document.objects.create(user=self.user, title="Doc B", file="documents/b.txt")

    def _make_run(self):
        run = AITaskRun.objects.create(user=self.user, task_type=AITaskRun.TaskType.SUMMARIZE, config={})
        AITaskRunDocument.objects.create(run=run, document=self.doc_a, role=AITaskRunDocument.Role.TARGET)
        AITaskRunDocument.objects.create(run=run, document=self.doc_b, role=AITaskRunDocument.Role.TARGET)
        return run

    def test_run_marked_failed_when_every_item_fails(self):
        run = self._make_run()

        with patch.object(ai_tasks_engine, "get_document_context_text", return_value={"text": "some content"}), \
             patch.object(ai_tasks_engine, "_call_llm_json", return_value=None):
            ai_tasks_engine.execute_run(run)

        run.refresh_from_db()
        self.assertEqual(run.status, AITaskRun.Status.FAILED)
        self.assertIn("2", run.error_message)
        self.assertEqual(
            run.results.filter(document__isnull=False, data__error=True).count(), 2
        )

    def test_run_marked_completed_with_a_partial_failure(self):
        run = self._make_run()
        success = {"summary": "ok", "key_points": [], "topics": []}

        with patch.object(ai_tasks_engine, "get_document_context_text", return_value={"text": "some content"}), \
             patch.object(ai_tasks_engine, "_call_llm_json", side_effect=[success, None]):
            ai_tasks_engine.execute_run(run)

        run.refresh_from_db()
        self.assertEqual(run.status, AITaskRun.Status.COMPLETED)
        self.assertIn("1 of 2", run.error_message)

    def test_run_marked_completed_when_everything_succeeds(self):
        run = self._make_run()
        success = {"summary": "ok", "key_points": [], "topics": []}

        with patch.object(ai_tasks_engine, "get_document_context_text", return_value={"text": "some content"}), \
             patch.object(ai_tasks_engine, "_call_llm_json", return_value=success):
            ai_tasks_engine.execute_run(run)

        run.refresh_from_db()
        self.assertEqual(run.status, AITaskRun.Status.COMPLETED)
        self.assertEqual(run.error_message, "")


class OrgPermissionServiceTests(TestCase):
    """
    Milestone 1/2 of the multi-tenancy build: Organization/
    OrganizationMembership models + org_permission_service's role/
    permission checks and privilege-escalation guards. No resource
    (Document, ...) is organization-aware yet - this is coverage for
    the isolation/authorization foundation alone, exercised against
    real DB rows the same way DocumentShareConstraintTests above does
    for its own model.
    """

    def setUp(self):
        from .services import org_permission_service as org_perm

        self.org_perm = org_perm

        self.org_type = OrganizationType.objects.create(slug="company_test", name="Company")
        self.org_a = Organization.objects.create(name="Org A", slug="org-a-test", org_type=self.org_type)
        self.org_b = Organization.objects.create(name="Org B", slug="org-b-test", org_type=self.org_type)

        self.owner = User.objects.create_user(username="org_owner", password="pw")
        self.member = User.objects.create_user(username="org_member", password="pw")
        self.outsider = User.objects.create_user(username="org_outsider", password="pw")
        self.suspended = User.objects.create_user(username="org_suspended", password="pw")

        OrganizationMembership.objects.create(organization=self.org_a, user=self.owner, role=org_perm.OWNER)
        OrganizationMembership.objects.create(organization=self.org_a, user=self.member, role=org_perm.MEMBER)
        OrganizationMembership.objects.create(
            organization=self.org_a, user=self.suspended, role=org_perm.MEMBER,
            status=OrganizationMembership.Status.SUSPENDED,
        )

    # ---- get_org_membership / cross-tenant isolation ----

    def test_get_org_membership_returns_none_for_non_member(self):
        self.assertIsNone(self.org_perm.get_org_membership(self.outsider, self.org_a))

    def test_get_org_membership_returns_none_for_suspended_membership(self):
        """A suspended membership must be indistinguishable from no membership at all - access revokes immediately."""
        self.assertIsNone(self.org_perm.get_org_membership(self.suspended, self.org_a))

    def test_get_org_membership_returns_none_across_organizations(self):
        """The core multi-tenancy guarantee: a real, active member of Org A has no membership row in Org B."""
        self.assertIsNone(self.org_perm.get_org_membership(self.owner, self.org_b))

    def test_get_org_membership_returns_row_for_active_member(self):
        membership = self.org_perm.get_org_membership(self.member, self.org_a)
        self.assertIsNotNone(membership)
        self.assertEqual(membership.role, self.org_perm.MEMBER)

    # ---- user_has_org_permission / role-rank gating ----

    def test_member_has_zero_org_management_permissions(self):
        """2026-09-06: a plain Member gets none of the organization-management surface at all - not even read-only organization.view."""
        for codename in self.org_perm.ORG_PERMISSION_MIN_ROLE:
            self.assertFalse(
                self.org_perm.user_has_org_permission(self.member, self.org_a, codename),
                f"Member unexpectedly holds '{codename}'",
            )
        self.assertEqual(self.org_perm.get_org_permission_codenames(self.member, self.org_a), [])

    def test_owner_holds_every_org_permission(self):
        for codename in self.org_perm.ORG_PERMISSION_MIN_ROLE:
            self.assertTrue(
                self.org_perm.user_has_org_permission(self.owner, self.org_a, codename),
                f"Owner unexpectedly lacks '{codename}'",
            )

    def test_outsider_has_no_org_permissions(self):
        self.assertFalse(self.org_perm.user_has_org_permission(self.outsider, self.org_a, "organization.view"))

    def test_unknown_codename_denies_rather_than_raises(self):
        self.assertFalse(self.org_perm.user_has_org_permission(self.owner, self.org_a, "not.a.real.codename"))

    # ---- privilege-escalation guards ----

    def test_owner_can_assign_any_role_including_owner(self):
        for role in (self.org_perm.MEMBER, self.org_perm.OWNER):
            self.assertTrue(self.org_perm.can_actor_assign_org_role(self.org_perm.OWNER, role))

    def test_member_can_assign_nothing(self):
        for role in (self.org_perm.MEMBER, self.org_perm.OWNER):
            self.assertFalse(self.org_perm.can_actor_assign_org_role(self.org_perm.MEMBER, role))

    def test_get_assignable_org_roles_matches_can_actor_assign(self):
        self.assertEqual(set(self.org_perm.get_assignable_org_roles(self.org_perm.OWNER)), {self.org_perm.MEMBER, self.org_perm.OWNER})
        self.assertEqual(set(self.org_perm.get_assignable_org_roles(self.org_perm.MEMBER)), set())

    def test_member_cannot_manage_a_peer_member_or_the_owner(self):
        self.assertFalse(self.org_perm.can_actor_manage_org_member(self.org_perm.MEMBER, self.org_perm.MEMBER))
        self.assertFalse(self.org_perm.can_actor_manage_org_member(self.org_perm.MEMBER, self.org_perm.OWNER))

    def test_owner_can_manage_a_member(self):
        self.assertTrue(self.org_perm.can_actor_manage_org_member(self.org_perm.OWNER, self.org_perm.MEMBER))

    # ---- is_last_org_owner ----

    def test_is_last_org_owner_true_for_sole_owner(self):
        self.assertTrue(self.org_perm.is_last_org_owner(self.owner, self.org_a))

    def test_is_last_org_owner_false_once_a_second_owner_exists(self):
        second_owner = User.objects.create_user(username="org_owner_2", password="pw")
        OrganizationMembership.objects.create(organization=self.org_a, user=second_owner, role=self.org_perm.OWNER)
        self.assertFalse(self.org_perm.is_last_org_owner(self.owner, self.org_a))

    def test_is_last_org_owner_false_for_non_owner(self):
        self.assertFalse(self.org_perm.is_last_org_owner(self.member, self.org_a))

    # ---- resolve_organization_context ----

    def test_resolve_organization_context_404s_for_unknown_slug(self):
        from django.http import Http404

        with self.assertRaises(Http404):
            self.org_perm.resolve_organization_context(self.member, "no-such-org-slug")

    def test_resolve_organization_context_404s_for_suspended_organization(self):
        from django.http import Http404

        self.org_a.status = Organization.Status.SUSPENDED
        self.org_a.save(update_fields=["status"])

        with self.assertRaises(Http404):
            self.org_perm.resolve_organization_context(self.member, self.org_a.slug)

    def test_resolve_organization_context_returns_none_membership_for_outsider(self):
        organization, membership = self.org_perm.resolve_organization_context(self.outsider, self.org_a.slug)
        self.assertEqual(organization, self.org_a)
        self.assertIsNone(membership)

    def test_resolve_organization_context_returns_membership_for_real_member(self):
        organization, membership = self.org_perm.resolve_organization_context(self.member, self.org_a.slug)
        self.assertEqual(organization, self.org_a)
        self.assertEqual(membership.role, self.org_perm.MEMBER)

    # ---- get_user_organizations / workspace switcher data ----

    def test_get_user_organizations_excludes_other_orgs_and_suspended_memberships(self):
        orgs = list(self.org_perm.get_user_organizations(self.member))
        self.assertEqual(len(orgs), 1)
        self.assertEqual(orgs[0].organization, self.org_a)

        self.assertEqual(list(self.org_perm.get_user_organizations(self.outsider)), [])
        self.assertEqual(list(self.org_perm.get_user_organizations(self.suspended)), [])


class OrganizationServiceTests(TestCase):
    """organization_service.py - create/update/suspend/delete, and the invariant that creating an organization always leaves exactly one Owner behind."""

    def setUp(self):
        self.org_type = OrganizationType.objects.create(slug="company_org_svc", name="Company")
        self.creator = User.objects.create_user(username="org_svc_creator", password="pw")

    def test_create_organization_makes_creator_owner(self):
        from .services import organization_service as org_svc
        from .services import org_permission_service as org_perm

        organization = org_svc.create_organization(name="Acme Inc", org_type=self.org_type, created_by=self.creator)

        membership = OrganizationMembership.objects.get(organization=organization, user=self.creator)
        self.assertEqual(membership.role, org_perm.OWNER)
        self.assertEqual(organization.slug, "acme-inc")

    def test_create_organization_auto_assigns_free_plan(self):
        from .models import Subscription
        from .services import organization_service as org_svc
        from .services.billing_service import FREE_PLAN_SLUG

        organization = org_svc.create_organization(name="Free Plan Co", org_type=self.org_type, created_by=self.creator)

        subscription = Subscription.objects.get(organization=organization)
        self.assertEqual(subscription.plan.slug, FREE_PLAN_SLUG)
        self.assertEqual(subscription.assigned_by, self.creator)

    def test_create_organization_generates_unique_slug_on_collision(self):
        from .services import organization_service as org_svc

        first = org_svc.create_organization(name="Acme Inc", org_type=self.org_type, created_by=self.creator)
        second_creator = User.objects.create_user(username="org_svc_creator_2", password="pw")
        second = org_svc.create_organization(name="Acme Inc", org_type=self.org_type, created_by=second_creator)

        self.assertNotEqual(first.slug, second.slug)
        self.assertTrue(second.slug.startswith("acme-inc-"))

    def test_create_organization_rejects_explicit_duplicate_slug(self):
        from .services import organization_service as org_svc

        org_svc.create_organization(name="Acme Inc", org_type=self.org_type, created_by=self.creator, slug="fixed-slug")
        other_creator = User.objects.create_user(username="org_svc_creator_3", password="pw")

        with self.assertRaises(org_svc.OrganizationServiceError):
            org_svc.create_organization(name="Different Name", org_type=self.org_type, created_by=other_creator, slug="fixed-slug")

    def test_suspend_blocks_context_resolution(self):
        from django.http import Http404

        from .services import org_permission_service as org_perm
        from .services import organization_service as org_svc

        organization = org_svc.create_organization(name="Suspend Test Co", org_type=self.org_type, created_by=self.creator)
        org_svc.suspend_organization(organization, self.creator)

        with self.assertRaises(Http404):
            org_perm.resolve_organization_context(self.creator, organization.slug)

    def test_delete_organization_cascades_memberships(self):
        from .services import organization_service as org_svc

        organization = org_svc.create_organization(name="Delete Test Co", org_type=self.org_type, created_by=self.creator)
        org_id = organization.id
        org_svc.delete_organization(organization, self.creator)

        self.assertFalse(Organization.objects.filter(id=org_id).exists())
        self.assertFalse(OrganizationMembership.objects.filter(organization_id=org_id).exists())


class OrgInvitationServiceTests(TestCase):
    """org_invitation_service.py - token generation, single-use/expiry, and the email-binding that stops a token being redeemed into the wrong account."""

    def setUp(self):
        from .services import org_permission_service as org_perm

        self.org_perm = org_perm
        self.org_type = OrganizationType.objects.create(slug="company_inv_svc", name="Company")
        self.organization = Organization.objects.create(name="Invite Co", slug="invite-co", org_type=self.org_type)
        self.owner = User.objects.create_user(username="inv_owner", password="pw")
        OrganizationMembership.objects.create(organization=self.organization, user=self.owner, role=org_perm.OWNER)

    def test_create_invitation_is_idempotent_for_same_pending_email(self):
        from .services.org_invitation_service import create_invitation

        first = create_invitation(self.organization, "invitee@example.com", self.org_perm.MEMBER, self.owner)
        second = create_invitation(self.organization, "INVITEE@example.com", self.org_perm.MEMBER, self.owner)

        self.assertEqual(first.id, second.id)
        self.assertEqual(OrganizationInvitation.objects.filter(organization=self.organization).count(), 1)

    def test_accept_invitation_creates_membership_with_invited_role(self):
        from .services.org_invitation_service import accept_invitation, create_invitation

        invitation = create_invitation(self.organization, "newmember@example.com", self.org_perm.MEMBER, self.owner)
        new_user = User.objects.create_user(username="newmember", password="pw", email="newmember@example.com")

        membership = accept_invitation(invitation.token, new_user)

        self.assertEqual(membership.role, self.org_perm.MEMBER)
        self.assertEqual(membership.organization, self.organization)
        invitation.refresh_from_db()
        self.assertEqual(invitation.status, OrganizationInvitation.Status.ACCEPTED)

    def test_accept_invitation_rejects_email_mismatch(self):
        from .services.org_invitation_service import InvitationError, accept_invitation, create_invitation

        invitation = create_invitation(self.organization, "intended@example.com", self.org_perm.MEMBER, self.owner)
        wrong_user = User.objects.create_user(username="wrong_person", password="pw", email="attacker@example.com")

        with self.assertRaises(InvitationError):
            accept_invitation(invitation.token, wrong_user)

        self.assertFalse(OrganizationMembership.objects.filter(organization=self.organization, user=wrong_user).exists())

    def test_accept_invitation_rejects_unknown_token(self):
        from .services.org_invitation_service import InvitationError, accept_invitation

        someone = User.objects.create_user(username="rando", password="pw", email="rando@example.com")
        with self.assertRaises(InvitationError):
            accept_invitation("not-a-real-token", someone)

    def test_accept_invitation_rejects_already_accepted_token(self):
        """Single-use: a token cannot be replayed after acceptance."""
        from .services.org_invitation_service import InvitationError, accept_invitation, create_invitation

        invitation = create_invitation(self.organization, "onceonly@example.com", self.org_perm.MEMBER, self.owner)
        first_user = User.objects.create_user(username="once_first", password="pw", email="onceonly@example.com")
        accept_invitation(invitation.token, first_user)

        second_user = User.objects.create_user(username="once_second", password="pw", email="onceonly@example.com")
        with self.assertRaises(InvitationError):
            accept_invitation(invitation.token, second_user)

    def test_accept_invitation_rejects_expired_token(self):
        from django.utils import timezone

        from .services.org_invitation_service import InvitationError, accept_invitation, create_invitation

        invitation = create_invitation(self.organization, "expired@example.com", self.org_perm.MEMBER, self.owner)
        invitation.expires_at = timezone.now() - timezone.timedelta(days=1)
        invitation.save(update_fields=["expires_at"])

        expired_user = User.objects.create_user(username="expired_user", password="pw", email="expired@example.com")
        with self.assertRaises(InvitationError):
            accept_invitation(invitation.token, expired_user)

        invitation.refresh_from_db()
        self.assertEqual(invitation.status, OrganizationInvitation.Status.EXPIRED)

    def test_revoke_invitation_prevents_later_acceptance(self):
        from .services.org_invitation_service import InvitationError, accept_invitation, create_invitation, revoke_invitation

        invitation = create_invitation(self.organization, "revoked@example.com", self.org_perm.MEMBER, self.owner)
        revoke_invitation(invitation, self.owner)

        revoked_user = User.objects.create_user(username="revoked_user", password="pw", email="revoked@example.com")
        with self.assertRaises(InvitationError):
            accept_invitation(invitation.token, revoked_user)

    def test_revoke_is_noop_for_already_accepted_invitation(self):
        from .services.org_invitation_service import accept_invitation, create_invitation, revoke_invitation

        invitation = create_invitation(self.organization, "already@example.com", self.org_perm.MEMBER, self.owner)
        accepted_user = User.objects.create_user(username="already_accepted", password="pw", email="already@example.com")
        accept_invitation(invitation.token, accepted_user)

        self.assertFalse(revoke_invitation(invitation, self.owner))
        invitation.refresh_from_db()
        self.assertEqual(invitation.status, OrganizationInvitation.Status.ACCEPTED)

    def test_send_org_invitation_email_task_delivers_real_email(self):
        """
        Calls the task directly rather than through task_runner.submit()
        - same "runs synchronously with no thread pool required"
        convention ProcessDocumentTaskTests uses - and lets
        send_templated_email actually render both templates (Django's
        test runner swaps in the locmem backend automatically), so a
        template syntax/context error would fail this test instead of
        only surfacing in production.
        """
        from django.core import mail

        from .services.org_invitation_service import create_invitation

        invitation = create_invitation(self.organization, "willbeinvited@example.com", self.org_perm.MEMBER, self.owner)

        tasks.send_org_invitation_email_task(invitation.id)

        self.assertEqual(len(mail.outbox), 1)
        sent = mail.outbox[0]
        self.assertEqual(sent.to, ["willbeinvited@example.com"])
        self.assertIn(self.organization.name, sent.subject)
        self.assertIn(invitation.token, sent.body)

    def test_send_org_invitation_email_task_skips_no_longer_pending_invitation(self):
        from django.core import mail

        from .services.org_invitation_service import create_invitation, revoke_invitation

        invitation = create_invitation(self.organization, "revokedbeforesend@example.com", self.org_perm.MEMBER, self.owner)
        revoke_invitation(invitation, self.owner)

        tasks.send_org_invitation_email_task(invitation.id)

        self.assertEqual(len(mail.outbox), 0)

    def test_send_org_invitation_email_task_handles_missing_invitation(self):
        """Never raises for a deleted/nonexistent invitation id - matches send_share_invite_email_task's own not-found handling."""
        tasks.send_org_invitation_email_task(999999)


class OrgMembershipServiceTests(TestCase):
    """org_membership_service.py - role changes/removal/suspension, including the last-Owner lockout guard end to end (not just the permission_service predicate)."""

    def setUp(self):
        from .services import org_permission_service as org_perm

        self.org_perm = org_perm
        self.org_type = OrganizationType.objects.create(slug="company_mem_svc", name="Company")
        self.organization = Organization.objects.create(name="Membership Co", slug="membership-co", org_type=self.org_type)

        self.owner_user = User.objects.create_user(username="mem_owner", password="pw")
        self.member_user = User.objects.create_user(username="mem_member", password="pw")

        self.owner_membership = OrganizationMembership.objects.create(organization=self.organization, user=self.owner_user, role=org_perm.OWNER)
        self.member_membership = OrganizationMembership.objects.create(organization=self.organization, user=self.member_user, role=org_perm.MEMBER)

    def test_owner_can_promote_member_to_owner(self):
        from .services.org_membership_service import update_member_role

        updated = update_member_role(self.organization, self.owner_membership, self.member_membership, self.org_perm.OWNER)
        self.assertEqual(updated.role, self.org_perm.OWNER)

    def test_member_cannot_change_anyones_role(self):
        from .services.org_membership_service import MembershipError, update_member_role

        with self.assertRaises(MembershipError):
            update_member_role(self.organization, self.member_membership, self.owner_membership, self.org_perm.MEMBER)

    def test_cannot_demote_last_owner(self):
        from .services.org_membership_service import MembershipError, update_member_role

        with self.assertRaises(MembershipError):
            update_member_role(self.organization, self.owner_membership, self.owner_membership, self.org_perm.MEMBER)

    def test_can_demote_owner_once_a_second_owner_exists(self):
        from .services.org_membership_service import update_member_role

        second_owner_user = User.objects.create_user(username="mem_owner_2", password="pw")
        second_owner_membership = OrganizationMembership.objects.create(organization=self.organization, user=second_owner_user, role=self.org_perm.OWNER)

        updated = update_member_role(self.organization, second_owner_membership, self.owner_membership, self.org_perm.MEMBER)
        self.assertEqual(updated.role, self.org_perm.MEMBER)

    def test_owner_can_remove_member(self):
        from .services.org_membership_service import remove_member

        remove_member(self.organization, self.owner_membership, self.member_membership)
        self.assertFalse(OrganizationMembership.objects.filter(id=self.member_membership.id).exists())

    def test_removed_members_account_is_deactivated(self):
        """
        2026-09-06 product decision: a company-registered member only
        ever exists inside that company (no personal workspace) - being
        removed blocks them from logging in at all, not just from that
        organization. Reversible: org_member_registration_service's
        re-registration path reactivates the same account rather than
        leaving it permanently dead (covered in
        OrgMemberRegistrationServiceTests).
        """
        from .services.org_membership_service import remove_member

        remove_member(self.organization, self.owner_membership, self.member_membership)
        self.member_user.refresh_from_db()
        self.assertFalse(self.member_user.is_active)

    def test_member_cannot_remove_the_owner(self):
        from .services.org_membership_service import MembershipError, remove_member

        with self.assertRaises(MembershipError):
            remove_member(self.organization, self.member_membership, self.owner_membership)

    def test_cannot_remove_last_owner(self):
        from .services.org_membership_service import MembershipError, remove_member

        with self.assertRaises(MembershipError):
            remove_member(self.organization, self.owner_membership, self.owner_membership)

    def test_cross_organization_membership_is_rejected(self):
        """A membership row from a DIFFERENT organization must never be actionable, even by that org's own Owner - closes an IDOR-shaped path where a client passes a foreign membership_id."""
        from .services.org_membership_service import MembershipError, remove_member

        other_org = Organization.objects.create(name="Other Co", slug="other-co-mem", org_type=self.org_type)
        other_user = User.objects.create_user(username="other_org_member", password="pw")
        other_membership = OrganizationMembership.objects.create(organization=other_org, user=other_user, role=self.org_perm.MEMBER)

        with self.assertRaises(MembershipError):
            remove_member(self.organization, self.owner_membership, other_membership)

    def test_suspend_then_reactivate_member(self):
        from .services.org_membership_service import set_member_status

        set_member_status(self.organization, self.owner_membership, self.member_membership, OrganizationMembership.Status.SUSPENDED)
        self.assertIsNone(self.org_perm.get_org_membership(self.member_user, self.organization))

        set_member_status(self.organization, self.owner_membership, self.member_membership, OrganizationMembership.Status.ACTIVE)
        self.assertIsNotNone(self.org_perm.get_org_membership(self.member_user, self.organization))


class OrganizationApiSecurityTests(TestCase):
    """
    End-to-end security tests through the real DRF endpoints (not just
    the service layer) - the explicit attack scenarios called for:
    cross-tenant access, IDOR, role escalation, unauthorized org
    switching, and forged organization identifiers. Uses Django's test
    Client with real session login, exactly how the React SPA talks to
    these endpoints.
    """

    def setUp(self):
        from .services import org_permission_service as org_perm
        from .services.organization_service import create_organization

        self.org_perm = org_perm
        self.org_type = OrganizationType.objects.create(slug="company_api_sec", name="Company")

        self.owner_a = User.objects.create_user(username="api_owner_a", password="pw12345")
        self.member_a = User.objects.create_user(username="api_member_a", password="pw12345")
        self.owner_b = User.objects.create_user(username="api_owner_b", password="pw12345")

        self.org_a = create_organization(name="API Org A", org_type=self.org_type, created_by=self.owner_a)
        OrganizationMembership.objects.create(organization=self.org_a, user=self.member_a, role=org_perm.MEMBER)

        self.org_b = create_organization(name="API Org B", org_type=self.org_type, created_by=self.owner_b)

    def test_outsider_cannot_view_organization_detail(self):
        self.client.login(username="api_owner_b", password="pw12345")
        response = self.client.get(f"/api/organizations/{self.org_a.slug}/")
        self.assertEqual(response.status_code, 403)

    def test_forged_org_slug_returns_404_not_a_data_leak(self):
        self.client.login(username="api_owner_a", password="pw12345")
        response = self.client.get("/api/organizations/this-org-does-not-exist/")
        self.assertEqual(response.status_code, 404)

    def test_member_cannot_invite(self):
        self.client.login(username="api_member_a", password="pw12345")
        response = self.client.post(
            f"/api/organizations/{self.org_a.slug}/invitations/",
            {"email": "someone@example.com", "role": "member"},
        )
        self.assertEqual(response.status_code, 403)

    def test_member_cannot_escalate_self_to_owner_via_invite(self):
        """A Member cannot invite (or otherwise obtain) an Owner-level role for themselves - role escalation attempt."""
        self.client.login(username="api_member_a", password="pw12345")
        response = self.client.post(
            f"/api/organizations/{self.org_a.slug}/invitations/",
            {"email": "escalate@example.com", "role": "org_owner"},
        )
        self.assertEqual(response.status_code, 403)

    def test_owner_a_cannot_manage_members_of_org_b(self):
        """Cross-tenant IDOR: Org A's Owner must not be able to act on Org B's membership even by hitting Org B's own correctly-scoped URL."""
        self.client.login(username="api_owner_a", password="pw12345")
        response = self.client.get(f"/api/organizations/{self.org_b.slug}/members/")
        self.assertEqual(response.status_code, 403)

    def test_member_id_from_another_org_is_rejected_even_with_valid_org_membership(self):
        """IDOR: Org A's Owner, acting within Org A's own (legitimately authorized) endpoint, passes a membership_id that belongs to Org B - must be rejected, not silently act cross-tenant."""
        org_b_membership_id = OrganizationMembership.objects.get(organization=self.org_b, user=self.owner_b).id

        self.client.login(username="api_owner_a", password="pw12345")
        response = self.client.post(
            f"/api/organizations/{self.org_a.slug}/members/action/",
            {"action": "remove", "membership_id": org_b_membership_id},
        )
        self.assertEqual(response.status_code, 404)
        self.assertTrue(OrganizationMembership.objects.filter(id=org_b_membership_id).exists())

    def test_suspended_organization_is_unreachable_even_by_its_owner(self):
        from .services.organization_service import suspend_organization

        suspend_organization(self.org_a, self.owner_a)

        self.client.login(username="api_owner_a", password="pw12345")
        response = self.client.get(f"/api/organizations/{self.org_a.slug}/")
        self.assertEqual(response.status_code, 404)

    def test_non_admin_cannot_reach_platform_organizations_view(self):
        self.client.login(username="api_member_a", password="pw12345")
        response = self.client.get("/api/admin/organizations/")
        self.assertEqual(response.status_code, 403)

    def test_platform_admin_can_view_all_organizations(self):
        """Role.has_permission() hardcodes Admin to always pass regardless of its M2M permission set (see Role.has_permission's docstring), so just creating the role - no seed_rbac/DEFAULT_PERMISSIONS needed - is sufficient here."""
        from .models import ADMIN_ROLE_SLUG, Role, UserRole

        admin_role, _ = Role.objects.get_or_create(slug=ADMIN_ROLE_SLUG, defaults={"name": "Admin", "is_system": True})
        UserRole.objects.create(user=self.owner_a, role=admin_role)

        self.client.login(username="api_owner_a", password="pw12345")
        response = self.client.get("/api/admin/organizations/")
        self.assertEqual(response.status_code, 200)
        slugs = {o["slug"] for o in response.json()["organizations"]}
        self.assertIn(self.org_a.slug, slugs)
        self.assertIn(self.org_b.slug, slugs)

    def test_invitation_accept_rejects_wrong_account_email(self):
        from .services.org_invitation_service import create_invitation

        invitation = create_invitation(self.org_a, "targeted@example.com", self.org_perm.MEMBER, self.owner_a)
        attacker = User.objects.create_user(username="api_attacker", password="pw12345", email="attacker@example.com")

        self.client.login(username="api_attacker", password="pw12345")
        response = self.client.post("/api/organizations/invitations/accept/", {"token": invitation.token})
        self.assertEqual(response.status_code, 400)
        self.assertFalse(OrganizationMembership.objects.filter(organization=self.org_a, user=attacker).exists())

    def test_anonymous_request_denied_everywhere(self):
        response = self.client.get(f"/api/organizations/{self.org_a.slug}/")
        self.assertIn(response.status_code, (401, 403))


class SuperAdminRoleTests(TestCase):
    """
    The "Super Admin" role (RAG/management/commands/seed_rbac.py) -
    unlike "admin", it has no Role.has_permission() bypass, so its
    access is entirely whatever seed_rbac actually attached to its M2M
    permission set. These tests seed it for real (call_command, not a
    hand-built Role) so a mistake in seed_rbac's SUPER_ADMIN_PERMISSIONS
    computation would actually be caught here.
    """

    def setUp(self):
        from django.core.management import call_command

        from .models import SUPER_ADMIN_ROLE_SLUG, Role, UserRole
        from .services.permission_service import SENSITIVE_PERMISSIONS

        call_command("seed_rbac")

        self.sensitive_permissions = SENSITIVE_PERMISSIONS
        self.super_admin_role = Role.objects.get(slug=SUPER_ADMIN_ROLE_SLUG)

        self.super_admin_user = User.objects.create_user(username="super_admin_user", password="pw12345")
        UserRole.objects.create(user=self.super_admin_user, role=self.super_admin_role)

        self.org_type = OrganizationType.objects.create(slug="company_super_admin", name="Company")
        self.other_owner = User.objects.create_user(username="super_admin_other_owner", password="pw12345")

        from .services.organization_service import create_organization

        self.organization = create_organization(name="Super Admin Test Co", org_type=self.org_type, created_by=self.other_owner)

    def test_super_admin_permission_set_is_every_permission_except_sensitive_ones(self):
        from .management.commands.seed_rbac import DEFAULT_PERMISSIONS

        expected = {c for c, _, _ in DEFAULT_PERMISSIONS} - self.sensitive_permissions
        actual = set(self.super_admin_role.permissions.values_list("codename", flat=True))

        self.assertEqual(actual, expected)
        for codename in self.sensitive_permissions:
            self.assertNotIn(codename, actual)

    def test_super_admin_can_view_and_manage_every_organization(self):
        self.client.login(username="super_admin_user", password="pw12345")

        response = self.client.get("/api/admin/organizations/")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["can_manage"])
        org = next(o for o in payload["organizations"] if o["slug"] == self.organization.slug)
        self.assertEqual(org["owner"], self.other_owner.username)
        self.assertEqual(org["size"], "")

        response = self.client.post(f"/api/admin/organizations/{self.organization.slug}/action/", {"action": "suspend"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "suspended")

    def test_super_admin_cannot_view_query_content(self):
        """
        Admin > Queries' content-detail endpoint (/api/admin/queries/
        <id>/detail/) was removed outright - not merely permission-
        gated - so this 404s for literally every user, Super Admin
        included, rather than 403ing only for a content-less one. See
        RAG.api.admin_queries_views' module docstring and
        AdminQueriesListPrivacyTests below for the metadata-only design
        this replaced it with.
        """
        query_log = QueryLog.objects.create(
            user=self.other_owner, question="What is in this document?", answer="Some answer.",
        )

        self.client.login(username="super_admin_user", password="pw12345")
        response = self.client.get(f"/api/admin/queries/{query_log.id}/detail/")
        self.assertEqual(response.status_code, 404)

    def test_super_admin_can_manage_another_users_ai_task_run(self):
        run = AITaskRun.objects.create(
            user=self.other_owner, task_type=AITaskRun.TaskType.ANALYZE, status=AITaskRun.Status.COMPLETED,
        )

        self.client.login(username="super_admin_user", password="pw12345")
        response = self.client.post(f"/api/ai-tasks/{run.id}/delete/")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(AITaskRun.objects.filter(id=run.id).exists())

    def test_super_admin_can_open_a_companys_stats_but_not_its_management_surface(self):
        """
        Regression test: clicking into a company from Admin > Companies
        used to 403 - Super Admin (like Admin) has organizations.manage
        but was never an actual OrganizationMembership row in any
        company it didn't personally create, and HasOrgPermission hard-
        required real membership. get_user_org_role() treats an
        organizations.manage holder as an OWNER of every organization
        for rank-comparison purposes, but (privacy fix) that no longer
        flows into user_has_org_permission()/get_org_permission_codenames() -
        those re-check real membership and, absent one, grant only
        BYPASS_ONLY_CODENAMES (organization.view). So platform oversight
        can still open a company's stats Overview, but members/billing/
        settings/audit-logs - the company's own management surface -
        now correctly 403 for a non-member, same as any other outsider.
        """
        self.client.login(username="super_admin_user", password="pw12345")

        response = self.client.get(f"/api/organizations/{self.organization.slug}/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["my_permissions"], ["organization.view"])

        response = self.client.get(f"/api/organizations/{self.organization.slug}/stats/")
        self.assertEqual(response.status_code, 200)

        response = self.client.get(f"/api/organizations/{self.organization.slug}/members/")
        self.assertEqual(response.status_code, 403)

        response = self.client.get(f"/api/organizations/{self.organization.slug}/billing/")
        self.assertEqual(response.status_code, 403)

        response = self.client.get(f"/api/organizations/{self.organization.slug}/audit-logs/")
        self.assertEqual(response.status_code, 403)

        # The separate platform-level action (suspend/delete) is untouched
        # by this restriction - it never goes through the org-scoped
        # codename table at all.
        response = self.client.post(f"/api/admin/organizations/{self.organization.slug}/action/", {"action": "suspend"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "suspended")

    def test_plain_member_still_cannot_open_a_company_they_do_not_belong_to(self):
        """The organizations.manage override must not leak to ordinary users - a non-privileged member of ONE company gets 403 on a DIFFERENT company they have no relationship to at all, exactly as before this fix."""
        outsider = User.objects.create_user(username="super_admin_outsider", password="pw12345")

        self.client.login(username="super_admin_outsider", password="pw12345")
        response = self.client.get(f"/api/organizations/{self.organization.slug}/")
        self.assertEqual(response.status_code, 403)


class DocumentOrganizationScopingTests(TestCase):
    """
    End-to-end HTTP coverage for wiring the React Documents API
    surface (RAG.api.documents_views) to org_permission_service.
    resolve_request_organization() - document_access_service's own
    organization-aware behavior already has unit coverage; this class
    confirms the *views* actually call it with the header-resolved
    organization, for real, through the same Django test Client the
    OrganizationApiSecurityTests class above uses.
    """

    def setUp(self):
        from .services import org_permission_service as org_perm
        from .services.organization_service import create_organization

        self.org_perm = org_perm

        self.permission, _ = Permission.objects.get_or_create(
            codename="pages.documents", defaults={"name": "Access Documents"},
        )
        self.role = Role.objects.create(slug="user_doc_scope", name="User")
        self.role.permissions.add(self.permission)

        self.org_type = OrganizationType.objects.create(slug="company_doc_scope", name="Company")

        self.user = User.objects.create_user(username="doc_scope_user", password="pw12345")
        UserRole.objects.create(user=self.user, role=self.role)

        self.other_org_owner = User.objects.create_user(username="doc_scope_other_org", password="pw12345")
        UserRole.objects.create(user=self.other_org_owner, role=self.role)

        self.org_a = create_organization(name="Doc Scope Org A", org_type=self.org_type, created_by=self.user)
        self.org_b = create_organization(name="Doc Scope Org B", org_type=self.org_type, created_by=self.other_org_owner)

        # create_organization() makes self.user account_type=COMPANY
        # (see its own docstring) - a real COMPANY account has no
        # Personal Workspace to fall back to (org_permission_service.
        # resolve_request_organization() defaults it straight to its
        # own organization even with no header), so any test that
        # actually needs "personal scope" behavior needs a genuinely
        # separate, org-less PERSONAL account instead of self.user.
        self.personal_user = User.objects.create_user(username="doc_scope_personal_user", password="pw12345")
        UserRole.objects.create(user=self.personal_user, role=self.role)

    def _upload(self, filename="report.txt", content=b"hello world", **extra):
        upload = SimpleUploadedFile(filename, content, content_type="text/plain")
        return self.client.post("/api/documents/upload/", {"document": upload}, **extra)

    def test_upload_without_org_header_lands_in_personal_workspace(self):
        self.client.login(username="doc_scope_personal_user", password="pw12345")
        response = self._upload()
        self.assertEqual(response.status_code, 201)
        document = Document.objects.get(id=response.json()["id"])
        self.assertIsNone(document.organization_id)

    def test_upload_with_org_header_tags_document_to_that_organization(self):
        self.client.login(username="doc_scope_user", password="pw12345")
        response = self._upload(HTTP_X_ORGANIZATION_SLUG=self.org_a.slug)
        self.assertEqual(response.status_code, 201)
        document = Document.objects.get(id=response.json()["id"])
        self.assertEqual(document.organization_id, self.org_a.id)

    def test_upload_with_org_header_for_org_the_user_is_not_a_member_of_is_rejected(self):
        self.client.login(username="doc_scope_user", password="pw12345")
        response = self._upload(HTTP_X_ORGANIZATION_SLUG=self.org_b.slug)
        self.assertEqual(response.status_code, 403)
        self.assertFalse(Document.objects.filter(user=self.user, organization=self.org_b).exists())

    def test_same_file_uploaded_to_personal_and_org_workspace_is_not_a_false_duplicate(self):
        """Duplicate detection is scoped per (user, organization), not globally - a Personal account and a Company account (or the same Company account's own two organizations, see check_duplicate()'s own docstring) uploading identical bytes into disjoint scopes must never trip each other's "already uploaded" check."""

        self.client.login(username="doc_scope_personal_user", password="pw12345")
        first = self._upload(content=b"identical bytes")
        self.assertEqual(first.status_code, 201)

        self.client.login(username="doc_scope_user", password="pw12345")
        second = self._upload(content=b"identical bytes", HTTP_X_ORGANIZATION_SLUG=self.org_a.slug)
        self.assertEqual(second.status_code, 201)

    def test_documents_list_shows_only_personal_documents_without_header(self):
        Document.objects.create(user=self.personal_user, title="Personal Doc", file=SimpleUploadedFile("p.txt", b"x"), file_hash="h1")
        Document.objects.create(user=self.user, title="Org Doc", file=SimpleUploadedFile("o.txt", b"y"), file_hash="h2", organization=self.org_a)

        self.client.login(username="doc_scope_personal_user", password="pw12345")
        response = self.client.get("/api/documents/")
        titles = {r["title"] for r in response.json()["results"]}
        self.assertIn("Personal Doc", titles)
        self.assertNotIn("Org Doc", titles)

    def test_documents_list_shows_only_active_organization_documents_with_header(self):
        Document.objects.create(user=self.user, title="Personal Doc", file=SimpleUploadedFile("p.txt", b"x"), file_hash="h1")
        Document.objects.create(user=self.user, title="Org Doc", file=SimpleUploadedFile("o.txt", b"y"), file_hash="h2", organization=self.org_a)

        self.client.login(username="doc_scope_user", password="pw12345")
        response = self.client.get("/api/documents/", HTTP_X_ORGANIZATION_SLUG=self.org_a.slug)
        titles = {r["title"] for r in response.json()["results"]}
        self.assertIn("Org Doc", titles)
        self.assertNotIn("Personal Doc", titles)

    def test_preview_cross_organization_document_is_not_found(self):
        """Org B's document must stay invisible to an Org A member both with no header (defaults to their own Org A - a Company account, per resolve_request_organization()) and with Org A's own header explicitly - never leaks across tenants regardless of which of those two equivalent ways the requester ends up scoped to Org A."""

        org_b_doc = Document.objects.create(
            user=self.other_org_owner, title="Org B Doc",
            file=SimpleUploadedFile("b.txt", b"z"), file_hash="h3", organization=self.org_b,
        )

        self.client.login(username="doc_scope_user", password="pw12345")

        response_no_header = self.client.get(f"/api/documents/{org_b_doc.id}/preview/")
        self.assertEqual(response_no_header.status_code, 404)

        response_org_a_header = self.client.get(f"/api/documents/{org_b_doc.id}/preview/", HTTP_X_ORGANIZATION_SLUG=self.org_a.slug)
        self.assertEqual(response_org_a_header.status_code, 404)

    def test_download_of_a_fellow_members_private_organization_document_is_not_found(self):
        """
        2026-09-05 product decision: organization membership grants
        ACTING inside the tenant, never blanket visibility into every
        document it owns - a member's upload defaults to private to
        them within their own company too, exactly like Personal
        Workspace already worked. This replaces a prior test that
        asserted the opposite (full org-wide transparency by default);
        see test_download_of_an_org_library_document_succeeds_for_a_
        fellow_member_who_did_not_upload_it below for the opt-in path
        that DOES make a document visible company-wide.
        """
        member = User.objects.create_user(username="doc_scope_member_a", password="pw12345")
        UserRole.objects.create(user=member, role=self.role)
        OrganizationMembership.objects.create(organization=self.org_a, user=member, role=self.org_perm.MEMBER)

        org_doc = Document.objects.create(
            user=self.user, title="Private In Org A",
            file=SimpleUploadedFile("private.txt", b"content"), file_hash="h4", organization=self.org_a,
        )

        self.client.login(username="doc_scope_member_a", password="pw12345")
        response = self.client.get(f"/api/documents/{org_doc.id}/download/", HTTP_X_ORGANIZATION_SLUG=self.org_a.slug)
        self.assertEqual(response.status_code, 404)

    def test_download_of_an_org_library_document_succeeds_for_a_fellow_member_who_did_not_upload_it(self):
        member = User.objects.create_user(username="doc_scope_member_b", password="pw12345")
        UserRole.objects.create(user=member, role=self.role)
        OrganizationMembership.objects.create(organization=self.org_a, user=member, role=self.org_perm.MEMBER)

        org_doc = Document.objects.create(
            user=self.user, title="Published To Org A Library",
            file=SimpleUploadedFile("library.txt", b"content"), file_hash="h5", organization=self.org_a,
            is_org_library=True,
        )

        self.client.login(username="doc_scope_member_b", password="pw12345")
        response = self.client.get(f"/api/documents/{org_doc.id}/download/", HTTP_X_ORGANIZATION_SLUG=self.org_a.slug)
        self.assertEqual(response.status_code, 200)

    def test_invalid_org_header_is_rejected_rather_than_silently_falling_back_to_personal(self):
        Document.objects.create(user=self.user, title="Personal Doc", file=SimpleUploadedFile("p.txt", b"x"), file_hash="h1")

        self.client.login(username="doc_scope_user", password="pw12345")
        response = self.client.get("/api/documents/", HTTP_X_ORGANIZATION_SLUG=self.org_b.slug)
        self.assertEqual(response.status_code, 403)

    def test_my_documents_list_inside_an_org_excludes_a_fellow_members_private_upload(self):
        """2026-09-05: 'My Documents' inside an organization is strictly this user's own uploads, same as Personal Workspace - never every document the tenant owns."""
        member = User.objects.create_user(username="doc_scope_member_c", password="pw12345")
        UserRole.objects.create(user=member, role=self.role)
        OrganizationMembership.objects.create(organization=self.org_a, user=member, role=self.org_perm.MEMBER)

        Document.objects.create(user=self.user, title="Owner's Private Doc", file=SimpleUploadedFile("owner.txt", b"x"), file_hash="h6", organization=self.org_a)

        self.client.login(username="doc_scope_member_c", password="pw12345")
        response = self.client.get("/api/documents/", HTTP_X_ORGANIZATION_SLUG=self.org_a.slug)
        titles = {r["title"] for r in response.json()["results"]}
        self.assertNotIn("Owner's Private Doc", titles)

    def test_org_library_page_excludes_a_different_organizations_library_documents(self):
        """
        Regression for a real cross-tenant leak found during the
        2026-09-05 audit: org_library_view had NO organization
        filtering at all (`Document.objects.filter(is_org_library=True)`,
        unconditionally global) - every organization's library was
        visible from every other organization's Org Library page.
        """
        Document.objects.create(
            user=self.other_org_owner, title="Org B Library Doc", file=SimpleUploadedFile("b_lib.txt", b"x"),
            file_hash="h7", organization=self.org_b, is_org_library=True,
        )
        Document.objects.create(
            user=self.user, title="Org A Library Doc", file=SimpleUploadedFile("a_lib.txt", b"x"),
            file_hash="h8", organization=self.org_a, is_org_library=True,
        )

        self.client.login(username="doc_scope_user", password="pw12345")
        response = self.client.get("/api/documents/org-library/", HTTP_X_ORGANIZATION_SLUG=self.org_a.slug)
        titles = {r["title"] for r in response.json()["results"]}
        self.assertIn("Org A Library Doc", titles)
        self.assertNotIn("Org B Library Doc", titles)

    def test_org_library_toggle_rejects_a_document_outside_the_active_organization(self):
        """The 'documents.manage_org_library' permission gate alone isn't enough - the target document must also belong to the actor's currently active organization, not just any organization the actor happens to hold that permission platform-wide."""
        permission, _ = Permission.objects.get_or_create(codename="documents.manage_org_library", defaults={"name": "Manage Org Library"})
        self.role.permissions.add(permission)

        foreign_doc = Document.objects.create(
            user=self.other_org_owner, title="Org B Doc To Steal", file=SimpleUploadedFile("steal.txt", b"x"),
            file_hash="h9", organization=self.org_b,
        )

        self.client.login(username="doc_scope_user", password="pw12345")
        response = self.client.post(f"/api/documents/org-library/{foreign_doc.id}/toggle/", HTTP_X_ORGANIZATION_SLUG=self.org_a.slug)
        self.assertEqual(response.status_code, 404)
        foreign_doc.refresh_from_db()
        self.assertFalse(foreign_doc.is_org_library)


class GraphConstructionOrganizationTaggingTests(TestCase):
    """
    build_graph_for_chunk() must tag every Entity/Relationship it
    creates with `chunk.document.organization` - never leave it NULL
    for an organization-owned document - so Personal Workspace and
    each organization keep independent (user, organization, name,
    entity_type) Entity rows instead of silently merging identically-
    named entities extracted from unrelated workspaces into one row.
    """

    def setUp(self):
        from .services.organization_service import create_organization

        self.user = User.objects.create_user(username="graph_org_tester", password="pw")
        self.org_type = OrganizationType.objects.create(slug="company_graph_org", name="Company")
        self.organization = create_organization(name="Graph Org", org_type=self.org_type, created_by=self.user)

        self.personal_document = Document.objects.create(user=self.user, title="Personal Doc", file="documents/p.txt")
        self.personal_chunk = DocumentChunk.objects.create(
            document=self.personal_document, content="Ada Lovelace worked with Charles Babbage.", chunk_number=0,
        )

        self.org_document = Document.objects.create(
            user=self.user, title="Org Doc", file="documents/o.txt", organization=self.organization,
        )
        self.org_chunk = DocumentChunk.objects.create(
            document=self.org_document, content="Ada Lovelace worked with Charles Babbage.", chunk_number=0,
        )

    def _extraction_result(self):
        return GraphExtractionResult(
            entities=[
                ExtractedEntity(name="Ada Lovelace", type="PERSON"),
                ExtractedEntity(name="Charles Babbage", type="PERSON"),
            ],
            relationships=[
                ExtractedRelationship(source="Ada Lovelace", relation="WORKED_WITH", target="Charles Babbage"),
            ],
        )

    def test_personal_document_chunk_creates_organization_null_entities(self):
        with patch("RAG.services.graph_service.extract_graph", return_value=self._extraction_result()):
            build_graph_for_chunk(self.personal_chunk, self.user)

        entities = Entity.objects.filter(user=self.user, name="ada lovelace")
        self.assertEqual(entities.count(), 1)
        self.assertIsNone(entities.first().organization_id)

        relationship = Relationship.objects.get(user=self.user)
        self.assertIsNone(relationship.organization_id)

    def test_org_document_chunk_creates_organization_tagged_entities(self):
        with patch("RAG.services.graph_service.extract_graph", return_value=self._extraction_result()):
            build_graph_for_chunk(self.org_chunk, self.user)

        entities = Entity.objects.filter(user=self.user, name="ada lovelace")
        self.assertEqual(entities.count(), 1)
        self.assertEqual(entities.first().organization_id, self.organization.id)

        relationship = Relationship.objects.get(user=self.user)
        self.assertEqual(relationship.organization_id, self.organization.id)

    def test_identically_named_entity_in_personal_and_org_workspace_does_not_merge(self):
        with patch("RAG.services.graph_service.extract_graph", return_value=self._extraction_result()):
            build_graph_for_chunk(self.personal_chunk, self.user)
            build_graph_for_chunk(self.org_chunk, self.user)

        # Two distinct Entity rows for the same (user, name, entity_type) -
        # one per workspace - not one row with a doubled mention_count.
        entities = Entity.objects.filter(user=self.user, name="ada lovelace").order_by("organization_id")
        self.assertEqual(entities.count(), 2)
        self.assertEqual([e.mention_count for e in entities], [1, 1])


class AskAIKnowledgeAITasksReportsOrganizationScopingTests(TestCase):
    """
    End-to-end HTTP coverage confirming Ask AI, Knowledge Graph,
    AI Tasks, and Reports all respect X-Organization-Slug the same way
    documents_views.py already does (DocumentOrganizationScopingTests
    above) - Personal Workspace and each organization must never bleed
    into each other's questions, entities, task runs, or report rows.
    """

    def setUp(self):
        from .services import org_permission_service as org_perm
        from .services.organization_service import create_organization

        self.org_perm = org_perm

        permissions = [
            "pages.documents", "pages.ask_ai", "pages.knowledge_base", "pages.ai_tasks", "pages.reports",
        ]
        self.role = Role.objects.create(slug="user_full_scope", name="User")
        for codename in permissions:
            permission, _ = Permission.objects.get_or_create(codename=codename, defaults={"name": codename})
            self.role.permissions.add(permission)

        self.org_type = OrganizationType.objects.create(slug="company_full_scope", name="Company")

        self.user = User.objects.create_user(username="full_scope_user", password="pw12345")
        UserRole.objects.create(user=self.user, role=self.role)

        self.organization = create_organization(name="Full Scope Org", org_type=self.org_type, created_by=self.user)
        self.other_organization = create_organization(
            name="Full Scope Org Two", org_type=self.org_type,
            created_by=User.objects.create_user(username="full_scope_other_owner", password="pw12345"),
        )
        # self.user belongs to BOTH organizations (Owner of one, Member
        # of the other) - see resolve_request_organization()'s
        # account-type-driven defaulting: a COMPANY account like
        # self.user has no Personal Workspace to fall back to at all,
        # so "workspace A's data is invisible while viewing workspace
        # B" has to be tested company-to-company, using a header for
        # BOTH sides, not "no header" as a stand-in for one of them.
        OrganizationMembership.objects.create(organization=self.other_organization, user=self.user, role=self.org_perm.MEMBER)

        # A THIRD organization self.user has no relationship to at all
        # - for the tests below that specifically need "an org this
        # user is not a member of", now that self.other_organization no
        # longer qualifies (self.user is a genuine Member there, above).
        self.foreign_organization = create_organization(
            name="Full Scope Foreign Org", org_type=self.org_type,
            created_by=User.objects.create_user(username="full_scope_foreign_owner", password="pw12345"),
        )

        # create_organization() makes self.user account_type=COMPANY -
        # a genuinely separate, org-less PERSONAL account is needed for
        # the handful of tests below that exercise real Personal
        # Workspace behavior (a plain no-header request from self.user
        # now defaults straight to self.organization, never Personal).
        self.personal_user = User.objects.create_user(username="full_scope_personal_user", password="pw12345")
        UserRole.objects.create(user=self.personal_user, role=self.role)

        self.personal_document = Document.objects.create(user=self.personal_user, title="Personal Doc", file="documents/p.txt", file_hash="fh1")
        self.org_document = Document.objects.create(
            user=self.user, title="Org Doc", file="documents/o.txt", file_hash="fh2", organization=self.organization,
        )
        self.other_org_document = Document.objects.create(
            user=self.other_organization.created_by, title="Other Org Doc", file="documents/oo.txt", file_hash="fh3", organization=self.other_organization,
        )

        self.client.login(username="full_scope_user", password="pw12345")

    # ---------------------------------------------------------------
    # Ask AI
    # ---------------------------------------------------------------

    def test_ask_context_documents_are_workspace_scoped(self):
        self.client.login(username="full_scope_personal_user", password="pw12345")
        personal_response = self.client.get("/api/ask/context/")
        personal_titles = {d["title"] for d in personal_response.json()["documents"]}
        self.assertIn("Personal Doc", personal_titles)
        self.assertNotIn("Org Doc", personal_titles)

        self.client.login(username="full_scope_user", password="pw12345")
        org_response = self.client.get("/api/ask/context/", HTTP_X_ORGANIZATION_SLUG=self.organization.slug)
        org_titles = {d["title"] for d in org_response.json()["documents"]}
        self.assertIn("Org Doc", org_titles)
        self.assertNotIn("Personal Doc", org_titles)

    def test_answer_question_tags_query_log_with_active_organization(self):
        chunks = [{"content": "x", "document": "Org Doc", "chunk_number": 0, "score": 0.1, "search_type": "vector"}]

        with patch.object(query_service, "retrieve_chunks", return_value=chunks), \
             patch.object(query_service, "generate_answer", return_value=("Answer [1].", {})):
            query_service.answer_question("A question", user=self.user, organization=self.organization)

        log = QueryLog.objects.get(user=self.user, question="A question")
        self.assertEqual(log.organization_id, self.organization.id)

    def test_ask_log_detail_is_not_visible_from_a_different_workspace(self):
        log = QueryLog.objects.create(
            user=self.user, question="Org-only question", answer="A", sources=[],
            search_method="Hybrid (Vector + BM25)", response_time_ms=1, confidence=50,
            organization=self.organization,
        )

        # self.user belongs to both organizations (see setUp) - viewing
        # from the OTHER one they're a genuine member of must still 404,
        # not just an org they aren't a member of at all (already
        # covered by test_invalid_org_header_rejected_on_every_newly_wired_surface).
        wrong_workspace = self.client.get(f"/api/ask/log/{log.id}/", HTTP_X_ORGANIZATION_SLUG=self.other_organization.slug)
        self.assertEqual(wrong_workspace.status_code, 404)

        right_workspace = self.client.get(f"/api/ask/log/{log.id}/", HTTP_X_ORGANIZATION_SLUG=self.organization.slug)
        self.assertEqual(right_workspace.status_code, 200)

    # ---------------------------------------------------------------
    # Knowledge Graph
    # ---------------------------------------------------------------

    def _make_entity(self, document, name, entity_type="PERSON"):
        chunk = DocumentChunk.objects.create(document=document, content=name, chunk_number=0)
        entity = Entity.objects.create(
            user=self.user, organization=document.organization, name=name.lower(),
            display_name=name, entity_type=entity_type, mention_count=1,
        )
        EntityMention.objects.create(entity=entity, chunk=chunk)
        return entity

    def test_knowledge_browse_topics_are_workspace_scoped(self):
        self._make_entity(self.personal_document, "Personal Person")
        self._make_entity(self.org_document, "Org Person")

        self.client.login(username="full_scope_personal_user", password="pw12345")
        personal_response = self.client.get("/api/knowledge/browse/")
        personal_names = {t["display_name"] for t in personal_response.json()["topics"]}
        self.assertIn("Personal Person", personal_names)
        self.assertNotIn("Org Person", personal_names)

        self.client.login(username="full_scope_user", password="pw12345")
        org_response = self.client.get("/api/knowledge/browse/", HTTP_X_ORGANIZATION_SLUG=self.organization.slug)
        org_names = {t["display_name"] for t in org_response.json()["topics"]}
        self.assertIn("Org Person", org_names)
        self.assertNotIn("Personal Person", org_names)

    def test_entity_detail_is_not_reachable_from_a_different_workspace(self):
        entity = self._make_entity(self.org_document, "Org Only Person")

        # self.user belongs to both organizations (see setUp) - viewing
        # from the OTHER one they're a genuine member of must still
        # 404, not just an org they aren't a member of at all.
        wrong_workspace = self.client.get(f"/api/knowledge/entities/{entity.id}/", HTTP_X_ORGANIZATION_SLUG=self.other_organization.slug)
        self.assertEqual(wrong_workspace.status_code, 404)

        right_workspace = self.client.get(f"/api/knowledge/entities/{entity.id}/", HTTP_X_ORGANIZATION_SLUG=self.organization.slug)
        self.assertEqual(right_workspace.status_code, 200)

    # ---------------------------------------------------------------
    # AI Tasks
    # ---------------------------------------------------------------

    def test_ai_task_create_rejects_a_document_outside_the_active_workspace(self):
        """A document from an org self.user is a genuine member of (self.other_organization - see setUp) submitted while Org A (self.organization, the default with no header) is active must still be rejected - membership in *some* organization is not the same as that document being in the currently active one."""

        with patch("RAG.services.task_runner.submit"):
            response = self.client.post(
                "/api/ai-tasks/create/",
                data=json.dumps({"task_type": "summarize", "document_ids": [self.other_org_document.id]}),
                content_type="application/json",
            )
        self.assertEqual(response.status_code, 400)

    def test_ai_task_create_tags_run_with_active_organization(self):
        with patch("RAG.services.task_runner.submit"):
            response = self.client.post(
                "/api/ai-tasks/create/",
                data=json.dumps({"task_type": "summarize", "document_ids": [self.org_document.id]}),
                content_type="application/json",
                HTTP_X_ORGANIZATION_SLUG=self.organization.slug,
            )
        self.assertEqual(response.status_code, 201)
        run = AITaskRun.objects.get(id=response.json()["id"])
        self.assertEqual(run.organization_id, self.organization.id)

    def test_ai_task_history_is_workspace_scoped(self):
        AITaskRun.objects.create(user=self.personal_user, task_type=AITaskRun.TaskType.SUMMARIZE, config={})
        AITaskRun.objects.create(user=self.user, task_type=AITaskRun.TaskType.SUMMARIZE, config={}, organization=self.organization)

        self.client.login(username="full_scope_personal_user", password="pw12345")
        personal_response = self.client.get("/api/ai-tasks/history/")
        self.assertEqual(personal_response.json()["count"], 1)

        self.client.login(username="full_scope_user", password="pw12345")
        org_response = self.client.get("/api/ai-tasks/history/", HTTP_X_ORGANIZATION_SLUG=self.organization.slug)
        self.assertEqual(org_response.json()["count"], 1)

    # ---------------------------------------------------------------
    # Select Documents picker (shared by Ask AI / AI Tasks)
    # ---------------------------------------------------------------

    def test_select_documents_search_is_workspace_scoped(self):
        self.client.login(username="full_scope_personal_user", password="pw12345")
        personal_response = self.client.get("/api/documents/select-dialog/search/")
        personal_titles = {d["title"] for d in personal_response.json()["results"]}
        self.assertIn("Personal Doc", personal_titles)
        self.assertNotIn("Org Doc", personal_titles)

        self.client.login(username="full_scope_user", password="pw12345")
        org_response = self.client.get("/api/documents/select-dialog/search/", HTTP_X_ORGANIZATION_SLUG=self.organization.slug)
        org_titles = {d["title"] for d in org_response.json()["results"]}
        self.assertIn("Org Doc", org_titles)
        self.assertNotIn("Personal Doc", org_titles)

    # ---------------------------------------------------------------
    # Reports
    # ---------------------------------------------------------------

    def test_reports_document_count_is_workspace_scoped(self):
        self.client.login(username="full_scope_personal_user", password="pw12345")
        personal_response = self.client.get("/api/reports/")
        self.assertEqual(personal_response.json()["document_count"], 1)

        self.client.login(username="full_scope_user", password="pw12345")
        org_response = self.client.get("/api/reports/", HTTP_X_ORGANIZATION_SLUG=self.organization.slug)
        self.assertEqual(org_response.json()["document_count"], 1)

    def test_invalid_org_header_rejected_on_every_newly_wired_surface(self):
        for path in (
            "/api/ask/context/",
            "/api/knowledge/browse/",
            "/api/ai-tasks/history/",
            "/api/documents/select-dialog/search/",
            "/api/reports/",
        ):
            response = self.client.get(path, HTTP_X_ORGANIZATION_SLUG=self.foreign_organization.slug)
            self.assertEqual(response.status_code, 403, f"{path} did not reject an org the user isn't a member of")

    # ---------------------------------------------------------------
    # Organization Overview stats (URL-slug-scoped, not header-scoped)
    # ---------------------------------------------------------------

    def test_organization_stats_counts_only_that_organizations_documents(self):
        response = self.client.get(f"/api/organizations/{self.organization.slug}/stats/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["document_count"], 1)

    def test_organization_stats_is_correct_even_while_a_different_workspace_is_active(self):
        """The active X-Organization-Slug header must never influence this URL-slug-scoped endpoint - viewing Org A's Overview while Org B is the active workspace must still describe Org A."""

        response = self.client.get(
            f"/api/organizations/{self.organization.slug}/stats/",
            HTTP_X_ORGANIZATION_SLUG=self.other_organization.slug,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["document_count"], 1)

    def test_organization_stats_rejects_a_non_member(self):
        response = self.client.get(f"/api/organizations/{self.foreign_organization.slug}/stats/")
        self.assertEqual(response.status_code, 403)


class LiveAskAIRetrievalOrganizationScopingTests(TransactionTestCase):
    """
    The one gap the rest of AskAIKnowledgeAITasksReportsOrganizationScopingTests
    leaves open: every test there either mocks retrieve_chunks() entirely
    or only exercises the accessible-documents list, never the real
    retrieve_chunks() -> bm25_search() pipeline end to end. This is that
    live test - and it needs TransactionTestCase, not TestCase:
    retrieve_chunks() runs vector/BM25/graph search concurrently via a
    ThreadPoolExecutor, and each worker thread opens its own DB
    connection. Plain TestCase wraps a test in one outer transaction
    that's rolled back, never committed - a different connection (i.e.
    every one of those worker threads) can never see fixture rows
    created inside that uncommitted transaction, so a real HTTP request
    through this pipeline would silently retrieve nothing at all under
    TestCase, regardless of any organization-scoping bug (confirmed by
    hand: a plain TestCase version of this exact test returns zero BM25
    results even for the ACTIVE organization's own document).
    TransactionTestCase actually commits, so background threads see it.
    """

    def setUp(self):
        from .services import org_permission_service as org_perm
        from .services.organization_service import create_organization

        self.role = Role.objects.create(slug="live_retrieval_scope", name="User")
        permission, _ = Permission.objects.get_or_create(codename="pages.ask_ai", defaults={"name": "pages.ask_ai"})
        self.role.permissions.add(permission)

        self.org_type = OrganizationType.objects.create(slug="live_retrieval_scope", name="Company")

        self.user = User.objects.create_user(username="live_retrieval_user", password="pw12345")
        UserRole.objects.create(user=self.user, role=self.role)
        self.organization = create_organization(name="Live Retrieval Org A", org_type=self.org_type, created_by=self.user)
        self.other_organization = create_organization(
            name="Live Retrieval Org B", org_type=self.org_type,
            created_by=User.objects.create_user(username="live_retrieval_other_owner", password="pw12345"),
        )
        # self.user is a genuine member of BOTH organizations (Owner of
        # one, Member of the other) - the same "membership in some org
        # isn't the same as that org being active" property the sibling
        # TestCase-based class exercises elsewhere in this file.
        OrganizationMembership.objects.create(organization=self.other_organization, user=self.user, role=org_perm.MEMBER)

        self.org_document = Document.objects.create(
            user=self.user, title="Org A Doc", file="documents/live_a.txt", file_hash="live_fh_a",
            organization=self.organization,
        )
        # Uploaded by self.user (a genuine Member of other_organization,
        # not its Owner) rather than other_organization.created_by -
        # this test is about cross-TENANT isolation (Org A vs Org B),
        # not within-org document privacy (a member's own upload is
        # always visible to them, in any organization they belong to;
        # see DocumentOrganizationScopingTests for the within-org
        # privacy-by-default coverage this class isn't testing).
        self.other_org_document = Document.objects.create(
            user=self.user, title="Org B Doc", file="documents/live_b.txt", file_hash="live_fh_b",
            organization=self.other_organization,
        )
        DocumentChunk.objects.create(document=self.org_document, content="The secret launch code is ALPHA-7.", chunk_number=0)
        DocumentChunk.objects.create(document=self.other_org_document, content="The secret launch code is BETA-9.", chunk_number=0)

        self.client.login(username="live_retrieval_user", password="pw12345")

    def test_live_ask_ai_retrieval_never_leaks_a_different_workspaces_content(self):
        with patch.object(query_service, "generate_answer", return_value=("Answer [1].", {})):
            org_a_response = self.client.post(
                "/api/ask/",
                data=json.dumps({"question": "What is the secret launch code?"}),
                content_type="application/json",
                HTTP_X_ORGANIZATION_SLUG=self.organization.slug,
            )
        self.assertEqual(org_a_response.status_code, 200)
        org_a_content = " ".join(s.get("content", "") for s in org_a_response.json()["sources"])
        self.assertIn("ALPHA-7", org_a_content)
        self.assertNotIn("BETA-9", org_a_content)

        with patch.object(query_service, "generate_answer", return_value=("Answer [1].", {})):
            org_b_response = self.client.post(
                "/api/ask/",
                data=json.dumps({"question": "What is the secret launch code?"}),
                content_type="application/json",
                HTTP_X_ORGANIZATION_SLUG=self.other_organization.slug,
            )
        self.assertEqual(org_b_response.status_code, 200)
        org_b_content = " ".join(s.get("content", "") for s in org_b_response.json()["sources"])
        self.assertIn("BETA-9", org_b_content)
        self.assertNotIn("ALPHA-7", org_b_content)


class AccountTypeSignupTests(TestCase):
    """
    Personal vs Company must be decided during signup itself, once,
    never through a workspace switcher afterward (see UserProfile.
    account_type's help_text). Covers both signup shapes end-to-end
    through the real /api/auth/signup/ endpoint, the organizations_view
    self-service-creation guard, and the invitation-accept flip for an
    invited employee who never saw the choice at all.
    """

    def setUp(self):
        self.org_type = OrganizationType.objects.create(slug="company_signup", name="Company")

    def _signup(self, **overrides):
        payload = {
            "full_name": "Signup Tester",
            "email": f"signup_{overrides.get('username', 'x')}@example.com",
            "username": "signup_default_user",
            "password": "correct horse battery staple 42",
            "confirm_password": "correct horse battery staple 42",
        }
        payload.update(overrides)
        return self.client.post("/api/auth/signup/", data=json.dumps(payload), content_type="application/json")

    def test_personal_signup_creates_a_personal_account_with_no_organization(self):
        response = self._signup(username="personal_signup_user", email="personal_signup@example.com", account_type="personal")
        self.assertEqual(response.status_code, 200)

        user = User.objects.get(username="personal_signup_user")
        self.assertEqual(user.profile.account_type, UserProfile.AccountType.PERSONAL)
        self.assertFalse(OrganizationMembership.objects.filter(user=user).exists())

    def test_omitting_account_type_defaults_to_personal(self):
        """The invitation-flow signup never sends account_type at all - it must default to personal, not silently grant any organization access."""

        response = self._signup(username="default_signup_user", email="default_signup@example.com")
        self.assertEqual(response.status_code, 200)

        user = User.objects.get(username="default_signup_user")
        self.assertEqual(user.profile.account_type, UserProfile.AccountType.PERSONAL)

    def test_company_signup_creates_the_organization_and_makes_the_signer_owner(self):
        response = self._signup(
            username="company_signup_user", email="company_signup@example.com",
            account_type="company", company_name="Acme Rockets", company_org_type=self.org_type.slug,
            company_size="11-50", company_industry="Aerospace",
        )
        self.assertEqual(response.status_code, 200)

        user = User.objects.get(username="company_signup_user")
        self.assertEqual(user.profile.account_type, UserProfile.AccountType.COMPANY)

        organization = Organization.objects.get(created_by=user)
        self.assertEqual(organization.name, "Acme Rockets")
        self.assertEqual(organization.size, "11-50")
        self.assertEqual(organization.industry, "Aerospace")

        membership = OrganizationMembership.objects.get(organization=organization, user=user)
        self.assertEqual(membership.role, OrganizationMembership.Role.OWNER)

    def test_company_signup_without_a_company_name_is_rejected(self):
        response = self._signup(
            username="company_no_name_user", email="company_no_name@example.com",
            account_type="company", company_org_type=self.org_type.slug,
        )
        self.assertEqual(response.status_code, 400)
        self.assertFalse(User.objects.filter(username="company_no_name_user").exists())

    def test_company_signup_with_an_invalid_org_type_is_rejected(self):
        response = self._signup(
            username="company_bad_type_user", email="company_bad_type@example.com",
            account_type="company", company_name="Acme", company_org_type="not-a-real-type",
        )
        self.assertEqual(response.status_code, 400)
        self.assertFalse(User.objects.filter(username="company_bad_type_user").exists())

    def test_personal_account_cannot_self_create_an_organization(self):
        user = User.objects.create_user(username="personal_no_create", password="pw12345")
        self.client.login(username="personal_no_create", password="pw12345")

        response = self.client.post(
            "/api/organizations/",
            data=json.dumps({"name": "Should Not Exist", "org_type": self.org_type.slug}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 403)
        self.assertFalse(Organization.objects.filter(name="Should Not Exist").exists())

    def test_accepting_an_invitation_converts_a_personal_account_to_company(self):
        from .services import org_permission_service as org_perm
        from .services.organization_service import create_organization
        from .services.org_invitation_service import accept_invitation, create_invitation

        owner = User.objects.create_user(username="invite_flip_owner", password="pw12345")
        organization = create_organization(name="Flip Co", org_type=self.org_type, created_by=owner)

        invitee = User.objects.create_user(username="invite_flip_user", password="pw12345", email="flip@example.com")
        self.assertEqual(invitee.profile.account_type, UserProfile.AccountType.PERSONAL)

        invitation = create_invitation(organization, "flip@example.com", org_perm.MEMBER, owner)
        accept_invitation(invitation.token, invitee)

        invitee.refresh_from_db()
        self.assertEqual(invitee.profile.account_type, UserProfile.AccountType.COMPANY)


class ResolveRequestOrganizationAccountTypeTests(TestCase):
    """org_permission_service.resolve_request_organization()'s account_type-driven defaulting - the server-side half of "no free Personal<->Company switching": a request header can never override what UserProfile.account_type + real membership rows say."""

    def setUp(self):
        from .services.organization_service import create_organization

        self.org_type = OrganizationType.objects.create(slug="company_resolve", name="Company")
        self.company_user = User.objects.create_user(username="resolve_company_user", password="pw12345")
        self.company_user.profile.account_type = UserProfile.AccountType.COMPANY
        self.company_user.profile.save(update_fields=["account_type"])
        self.organization = create_organization(name="Resolve Co", org_type=self.org_type, created_by=self.company_user)

        self.personal_user = User.objects.create_user(username="resolve_personal_user", password="pw12345")

    def _request_for(self, user, headers=None):
        request = RequestFactory().get("/", **(headers or {}))
        request.user = user
        return request

    def test_company_account_defaults_to_their_own_organization_without_a_header(self):
        from .services.org_permission_service import resolve_request_organization

        organization, membership = resolve_request_organization(self._request_for(self.company_user))
        self.assertEqual(organization, self.organization)
        self.assertIsNotNone(membership)

    def test_personal_account_with_a_header_for_an_org_it_does_not_belong_to_is_rejected(self):
        """Real membership, not account_type alone, is the authorization source once a header is present - a PERSONAL account naming a real org it never joined gets the same PermissionDenied any non-member would, not a silent fallback to Personal Workspace."""

        from django.core.exceptions import PermissionDenied
        from .services.org_permission_service import resolve_request_organization

        with self.assertRaises(PermissionDenied):
            resolve_request_organization(
                self._request_for(self.personal_user, {"HTTP_X_ORGANIZATION_SLUG": self.organization.slug})
            )

    def test_personal_account_with_no_header_never_defaults_into_an_organization(self):
        """The no-header DEFAULT is where account_type actually matters: even if a PERSONAL-flagged account somehow holds real membership (e.g. stale pre-migration data), a bare request must still default to Personal Workspace, never silently to that organization."""

        from .services.org_permission_service import resolve_request_organization

        OrganizationMembership.objects.create(
            organization=self.organization, user=self.personal_user, role=OrganizationMembership.Role.MEMBER,
        )

        organization, membership = resolve_request_organization(self._request_for(self.personal_user))
        self.assertIsNone(organization)
        self.assertIsNone(membership)

    def test_company_account_with_no_organizations_left_degrades_to_none_rather_than_erroring(self):
        from .services.org_permission_service import resolve_request_organization

        lonely_company_user = User.objects.create_user(username="resolve_lonely_company", password="pw12345")
        lonely_company_user.profile.account_type = UserProfile.AccountType.COMPANY
        lonely_company_user.profile.save(update_fields=["account_type"])

        organization, membership = resolve_request_organization(self._request_for(lonely_company_user))
        self.assertIsNone(organization)
        self.assertIsNone(membership)


class BillingTests(TestCase):
    """
    Billing/Plans/Usage Limits - internal bookkeeping only, hard-block
    enforcement. Covers: no Subscription means unlimited (new-org
    default), a Plan's own max_queries_per_month is informational only
    (2026-09-08 redesign - AI credits are the sole org-wide
    spend-metering currency, see billing_service.py's module
    docstring) while an Owner-set per-member cap still hard-blocks that
    one member, an Owner cannot reach any Plan-writing endpoint, and
    period rollover resets usage.
    """

    def setUp(self):
        from .services.organization_service import create_organization
        from .services import org_permission_service as org_perm

        self.org_perm = org_perm
        self.org_type = OrganizationType.objects.create(slug="billing_test_org_type", name="Company")

        self.owner = User.objects.create_user(username="billing_owner", password="pw12345")
        self.organization = create_organization(name="Billing Test Org", org_type=self.org_type, created_by=self.owner)

        self.plan = Plan.objects.create(name="Starter", slug="starter", max_queries_per_month=2, max_seats=5)

    def test_organization_with_no_subscription_is_never_blocked(self):
        from .services import billing_service

        # No UsageLimitExceeded raised, no matter how much usage exists.
        for _ in range(10):
            QueryLog.objects.create(
                user=self.owner, organization=self.organization, question="q", answer="a",
                search_method="Hybrid (Vector + BM25)", response_time_ms=1, confidence=50,
            )
        billing_service.check_query_limit_for(self.organization, self.owner)  # must not raise
        usage = billing_service.get_organization_usage(self.organization)
        self.assertTrue(usage["unlimited"])

    def test_plan_query_count_is_informational_only_not_enforced(self):
        """A Plan's own max_queries_per_month no longer raises on its own once exceeded - AI credits are the organization's sole spend-metering currency now (see billing_service.py's module docstring). The count is still reported by get_organization_usage(), just never a block by itself."""
        from .services import billing_service

        billing_service.assign_plan(self.organization, self.plan, self.owner)

        for _ in range(5):  # plan's max_queries_per_month=2, deliberately exceeded
            QueryLog.objects.create(
                user=self.owner, organization=self.organization, question="q", answer="a",
                search_method="Hybrid (Vector + BM25)", response_time_ms=1, confidence=50,
            )

        billing_service.check_query_limit_for(self.organization, self.owner)  # must not raise - no member-specific cap set
        usage = billing_service.get_organization_usage(self.organization)
        self.assertEqual(usage["queries_used"], 5)

    def test_member_query_limit_blocks_only_that_member(self):
        """The Owner-set per-member sub-quota (OrganizationMembership.max_queries_per_month) remains a real, count-based hard block, independent of the org-wide credit balance and of the Plan's now-informational max_queries_per_month."""
        from .services import billing_service
        from .services.org_member_limits_service import set_member_usage_limits

        member = User.objects.create_user(username="billing_member", password="pw12345")
        member_membership = OrganizationMembership.objects.create(
            organization=self.organization, user=member, role=self.org_perm.MEMBER,
        )
        owner_membership = OrganizationMembership.objects.get(organization=self.organization, user=self.owner)
        set_member_usage_limits(self.organization, owner_membership, member_membership, 1, None)

        QueryLog.objects.create(
            user=member, organization=self.organization, question="q", answer="a",
            search_method="Hybrid (Vector + BM25)", response_time_ms=1, confidence=50,
        )

        with self.assertRaises(billing_service.UsageLimitExceeded) as ctx:
            billing_service.check_query_limit_for(self.organization, member)
        self.assertEqual(ctx.exception.limit_type, "member_queries")

        # The Owner, who has no member-specific cap, is unaffected.
        billing_service.check_query_limit_for(self.organization, self.owner)

    def test_member_limit_cannot_exceed_the_organizations_own_plan_limit(self):
        from .services import billing_service
        from .services.org_member_limits_service import MemberLimitError, set_member_usage_limits

        billing_service.assign_plan(self.organization, self.plan, self.owner)  # plan's max_queries_per_month=2

        member = User.objects.create_user(username="billing_member_2", password="pw12345")
        member_membership = OrganizationMembership.objects.create(
            organization=self.organization, user=member, role=self.org_perm.MEMBER,
        )
        owner_membership = OrganizationMembership.objects.get(organization=self.organization, user=self.owner)

        with self.assertRaises(MemberLimitError):
            set_member_usage_limits(self.organization, owner_membership, member_membership, 5, None)

    def test_seat_limit_blocks_new_invitation(self):
        from .services import billing_service
        from .services.org_invitation_service import create_invitation

        low_seat_plan = Plan.objects.create(name="Tiny", slug="tiny", max_seats=1)
        billing_service.assign_plan(self.organization, low_seat_plan, self.owner)
        # self.owner's own OWNER membership already counts as 1 seat.

        with self.assertRaises(billing_service.UsageLimitExceeded) as ctx:
            create_invitation(self.organization, "newperson@example.com", self.org_perm.MEMBER, self.owner)
        self.assertEqual(ctx.exception.limit_type, "seats")

    def test_owner_cannot_reach_any_plan_writing_endpoint(self):
        self.client.login(username="billing_owner", password="pw12345")

        response = self.client.get("/api/admin/billing/plans/")
        self.assertEqual(response.status_code, 403)

        response = self.client.post(
            "/api/admin/billing/plans/", data=json.dumps({"name": "Whatever"}), content_type="application/json",
        )
        self.assertEqual(response.status_code, 403)

        response = self.client.post(
            f"/api/admin/billing/organizations/{self.organization.slug}/assign-plan/",
            data=json.dumps({"plan_id": self.plan.id}), content_type="application/json",
        )
        self.assertEqual(response.status_code, 403)

    def test_owner_can_view_own_organization_billing_read_only(self):
        from .services import billing_service

        billing_service.assign_plan(self.organization, self.plan, self.owner)

        self.client.login(username="billing_owner", password="pw12345")
        response = self.client.get(f"/api/organizations/{self.organization.slug}/billing/")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["unlimited"])
        self.assertEqual(response.json()["plan"]["slug"], "starter")

    def test_period_rollover_resets_usage(self):
        from datetime import timedelta

        from django.utils import timezone

        from .services import billing_service

        subscription = billing_service.assign_plan(self.organization, self.plan, self.owner)
        # Force the period to have already lapsed.
        subscription.current_period_start = timezone.now() - timedelta(days=60)
        subscription.current_period_end = timezone.now() - timedelta(days=30)
        subscription.save(update_fields=["current_period_start", "current_period_end"])

        # A QueryLog from the OLD (lapsed) period must not count toward the new one.
        old_log = QueryLog.objects.create(
            user=self.owner, organization=self.organization, question="old", answer="a",
            search_method="Hybrid (Vector + BM25)", response_time_ms=1, confidence=50,
        )
        old_log.created_at = timezone.now() - timedelta(days=45)
        old_log.save(update_fields=["created_at"])

        usage = billing_service.get_organization_usage(self.organization)
        self.assertEqual(usage["queries_used"], 0)
        # The rolled-forward period must actually contain "now" - advancing
        # in whole-month steps from a long-lapsed period doesn't imply
        # landing within the last day of it, just that it no longer excludes it.
        now = timezone.now()
        self.assertLessEqual(usage["period_start"], now)
        self.assertGreater(usage["period_end"], now)

    def test_plan_cannot_be_hard_deleted_while_assigned(self):
        from django.db.models import ProtectedError

        from .services import billing_service

        billing_service.assign_plan(self.organization, self.plan, self.owner)
        with self.assertRaises(ProtectedError):
            self.plan.delete()


class MemberActionPermissionRegressionTests(TestCase):
    """
    Regression coverage for organization_member_action_view's
    permission gate. Originally written 2026-09-05 for a bug where a
    Manager could remove/suspend/reassign a Member (fixed by requiring
    ADMIN rank); updated 2026-09-06 when Admin/Manager were removed
    entirely (product decision: only Owner and Member exist now, and
    Owner holds the ENTIRE organization-management surface - Member
    holds none of it, not even read-only access to the org tab at
    all). The underlying invariant these tests protect is unchanged:
    only an Owner may act on a member.
    """

    def setUp(self):
        from .services import org_permission_service as org_perm
        from .services.organization_service import create_organization

        self.org_perm = org_perm
        self.org_type = OrganizationType.objects.create(slug="member_action_org_type", name="Company")

        self.owner = User.objects.create_user(username="ma_owner", password="pw12345")
        self.organization = create_organization(name="Member Action Org", org_type=self.org_type, created_by=self.owner)

        self.member = User.objects.create_user(username="ma_member", password="pw12345")
        self.member_membership = OrganizationMembership.objects.create(
            organization=self.organization, user=self.member, role=org_perm.MEMBER,
        )

    def _action(self, actor_username, action, **extra):
        self.client.login(username=actor_username, password="pw12345")
        return self.client.post(
            f"/api/organizations/{self.organization.slug}/members/action/",
            {"action": action, "membership_id": self.member_membership.id, **extra},
        )

    def test_member_cannot_remove_a_member(self):
        second_member = User.objects.create_user(username="ma_member_2", password="pw12345")
        second_membership = OrganizationMembership.objects.create(organization=self.organization, user=second_member, role=self.org_perm.MEMBER)

        self.client.login(username="ma_member", password="pw12345")
        response = self.client.post(
            f"/api/organizations/{self.organization.slug}/members/action/",
            {"action": "remove", "membership_id": second_membership.id},
        )
        self.assertEqual(response.status_code, 403)
        self.assertTrue(OrganizationMembership.objects.filter(id=second_membership.id).exists())

    def test_member_cannot_suspend_a_member(self):
        response = self._action("ma_member", "suspend")
        self.assertEqual(response.status_code, 403)

    def test_member_cannot_change_a_members_role(self):
        response = self._action("ma_member", "update_role", role=self.org_perm.OWNER)
        self.assertEqual(response.status_code, 403)
        self.member_membership.refresh_from_db()
        self.assertEqual(self.member_membership.role, self.org_perm.MEMBER)

    def test_owner_can_remove_a_member(self):
        response = self._action("ma_owner", "remove")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(OrganizationMembership.objects.filter(id=self.member_membership.id).exists())

    def test_member_cannot_view_the_member_roster(self):
        """2026-09-06: Member holds none of the org-management surface, including the previously-allowed read-only roster view."""
        self.client.login(username="ma_member", password="pw12345")
        response = self.client.get(f"/api/organizations/{self.organization.slug}/members/")
        self.assertEqual(response.status_code, 403)

    def test_member_cannot_view_audit_logs(self):
        self.client.login(username="ma_member", password="pw12345")
        response = self.client.get(f"/api/organizations/{self.organization.slug}/audit-logs/")
        self.assertEqual(response.status_code, 403)

    def test_member_cannot_view_organization_overview(self):
        self.client.login(username="ma_member", password="pw12345")
        response = self.client.get(f"/api/organizations/{self.organization.slug}/")
        self.assertEqual(response.status_code, 403)

    def test_owner_can_view_audit_logs(self):
        self.client.login(username="ma_owner", password="pw12345")
        response = self.client.get(f"/api/organizations/{self.organization.slug}/audit-logs/")
        self.assertEqual(response.status_code, 200)


class DocumentOrgAuditLogTests(TestCase):
    """
    Regression coverage for the audit-log gap found during this audit:
    an organization-owned document's create/delete/archive events used
    to only reach the platform-wide ActivityLog, never that
    organization's own OrganizationAuditLog - so an Owner/Admin looking
    at their org's Audit Log page never saw document activity at all,
    even though the audit pipeline itself (model/service/view/frontend)
    was already wired correctly.
    """

    def setUp(self):
        from .services.organization_service import create_organization

        self.org_type = OrganizationType.objects.create(slug="doc_audit_org_type", name="Company")
        self.owner = User.objects.create_user(username="doc_audit_owner", password="pw12345")
        self.organization = create_organization(name="Doc Audit Org", org_type=self.org_type, created_by=self.owner)

        # HasPagePermission("pages.documents") gates the document
        # endpoints below - the test DB has no seed_rbac data by
        # default, so (matching DocumentOrganizationScopingTests'
        # established pattern above) the permission/role/assignment
        # are created explicitly rather than assumed to pre-exist.
        permission, _ = Permission.objects.get_or_create(codename="pages.documents", defaults={"name": "Access Documents"})
        role = Role.objects.create(slug="doc_audit_user_role", name="User")
        role.permissions.add(permission)
        UserRole.objects.create(user=self.owner, role=role)

    def test_uploading_an_org_document_is_recorded_in_the_org_audit_log(self):
        self.client.login(username="doc_audit_owner", password="pw12345")
        upload = SimpleUploadedFile("report.txt", b"hello world", content_type="text/plain")

        response = self.client.post(
            "/api/documents/upload/", {"document": upload},
            HTTP_X_ORGANIZATION_SLUG=self.organization.slug,
        )
        self.assertEqual(response.status_code, 201)

        entry = self.organization.audit_logs.filter(action="document.uploaded").first()
        self.assertIsNotNone(entry)
        self.assertEqual(entry.actor, self.owner)
        self.assertEqual(entry.metadata.get("document_id"), response.json()["id"])

    def test_deleting_an_org_document_is_recorded_in_the_org_audit_log(self):
        document = Document.objects.create(
            user=self.owner, title="Doomed", file="documents/doomed.txt", file_hash="doc_audit_fh1", organization=self.organization,
        )

        self.client.login(username="doc_audit_owner", password="pw12345")
        response = self.client.delete(f"/api/documents/{document.id}/")
        self.assertEqual(response.status_code, 204)

        entry = self.organization.audit_logs.filter(action="document.deleted").first()
        self.assertIsNotNone(entry)
        self.assertEqual(entry.metadata.get("document_id"), document.id)

    def test_personal_document_activity_never_reaches_any_org_audit_log(self):
        """
        A genuinely PERSONAL account (never a member of any
        organization - a COMPANY account like self.owner always
        resolves to SOME organization by default, per
        resolve_request_organization(), so it can't exercise this
        path) uploading a document must produce zero
        OrganizationAuditLog rows anywhere.
        """

        permission = Permission.objects.get(codename="pages.documents")
        role = Role.objects.create(slug="doc_audit_personal_role", name="Personal User")
        role.permissions.add(permission)
        personal_user = User.objects.create_user(username="doc_audit_personal", password="pw12345")
        UserRole.objects.create(user=personal_user, role=role)

        self.client.login(username="doc_audit_personal", password="pw12345")
        upload = SimpleUploadedFile("personal.txt", b"just for me", content_type="text/plain")
        response = self.client.post("/api/documents/upload/", {"document": upload})
        self.assertEqual(response.status_code, 201)

        document = Document.objects.get(id=response.json()["id"])
        self.assertIsNone(document.organization)
        # setUp()'s create_organization() itself logs "organization.created" -
        # this asserts no ADDITIONAL (specifically document-related) row
        # appeared, not that the log is empty.
        self.assertFalse(self.organization.audit_logs.filter(action__startswith="document.").exists())


class DashboardWorkspaceIsolationTests(TestCase):
    """
    Regression coverage for the dashboard/stats isolation gap found
    during this audit: stats_service.py's aggregates took `user` only,
    with no `organization` parameter at all - so Document.objects.
    filter(user=user) (a user's Personal AND every organization they've
    ever uploaded to, merged with no way to tell them apart) backed the
    dashboard regardless of which workspace was actually active.
    """

    def setUp(self):
        from .services.organization_service import create_organization

        self.org_type = OrganizationType.objects.create(slug="dash_iso_org_type", name="Company")

        self.user = User.objects.create_user(username="dash_iso_user", password="pw12345")
        self.org_a = create_organization(name="Dash Iso Org A", org_type=self.org_type, created_by=self.user)

        other_owner = User.objects.create_user(username="dash_iso_other_owner", password="pw12345")
        self.org_b = create_organization(name="Dash Iso Org B", org_type=self.org_type, created_by=other_owner)
        OrganizationMembership.objects.create(organization=self.org_b, user=self.user, role=OrganizationMembership.Role.MEMBER)

        self.personal_doc = Document.objects.create(
            user=self.user, title="Personal Doc", file="documents/p.txt", file_hash="dash_iso_fh1", organization=None,
        )
        self.org_a_doc = Document.objects.create(
            user=self.user, title="Org A Doc", file="documents/a.txt", file_hash="dash_iso_fh2", organization=self.org_a,
        )
        self.org_b_doc = Document.objects.create(
            user=other_owner, title="Org B Doc", file="documents/b.txt", file_hash="dash_iso_fh3", organization=self.org_b,
        )

    def test_personal_dashboard_excludes_every_organizations_documents(self):
        from .services.stats_service import get_dashboard_stats

        stats = get_dashboard_stats(self.user, organization=None)
        self.assertEqual(stats["total_documents"], 1)

    def test_organization_a_dashboard_excludes_personal_and_organization_b(self):
        from .services.stats_service import get_dashboard_stats

        stats = get_dashboard_stats(self.user, organization=self.org_a)
        self.assertEqual(stats["total_documents"], 1)

    def test_organization_b_dashboard_excludes_organization_a(self):
        """The same user, viewing a DIFFERENT organization they also belong to, must not see Org A's document at all."""
        from .services.stats_service import get_dashboard_stats

        stats = get_dashboard_stats(self.user, organization=self.org_b)
        self.assertEqual(stats["total_documents"], 1)

    def test_organization_dashboard_is_organization_wide_not_just_the_viewers_own_uploads(self):
        """
        get_dashboard_stats() itself still defaults to whole-org
        (scope_to_own=False) when called directly - the Owner-vs-Member
        split (see test_dashboard_api_reflects_the_header_scoped_workspace
        below) is decided by dashboard_view, not by this function's
        default. org_b_doc was uploaded by a different user than
        self.user, so this only holds when the whole org is in scope.
        """
        from .services.stats_service import get_dashboard_stats

        stats = get_dashboard_stats(self.user, organization=self.org_b)
        self.assertEqual(stats["last_upload"].id, self.org_b_doc.id)

    def test_dashboard_api_reflects_the_header_scoped_workspace(self):
        """
        self.user is Owner of org_a (created it) but only a MEMBER of
        org_b. dashboard_view now scopes a Member's numbers to their
        own rows within the org - org_b's only document was uploaded
        by other_owner, not self.user, so the Member view of org_b
        must show 0, while the Owner view of org_a still shows the
        whole organization's 1 document.
        """
        self.client.login(username="dash_iso_user", password="pw12345")

        personal_response = self.client.get("/api/dashboard/")
        self.assertEqual(personal_response.json()["stats"]["total_documents"], 1)

        org_a_response = self.client.get("/api/dashboard/", HTTP_X_ORGANIZATION_SLUG=self.org_a.slug)
        self.assertEqual(org_a_response.json()["stats"]["total_documents"], 1)

        org_b_response = self.client.get("/api/dashboard/", HTTP_X_ORGANIZATION_SLUG=self.org_b.slug)
        self.assertEqual(org_b_response.json()["stats"]["total_documents"], 0)

    def test_dashboard_api_owner_still_sees_whole_organization(self):
        """The org_b Owner (other_owner), viewing the same org_b workspace, must still see the whole-org number a Member does not."""
        self.client.login(username="dash_iso_other_owner", password="pw12345")

        org_b_response = self.client.get("/api/dashboard/", HTTP_X_ORGANIZATION_SLUG=self.org_b.slug)
        self.assertEqual(org_b_response.json()["stats"]["total_documents"], 1)


class MyActivitySummaryTests(TestCase):
    """
    stats_service.get_my_activity_summary() - the Owner's personal
    "Your Activity" slice surfaced alongside organization_stats_view's
    whole-org numbers. Must count only the requesting user's own rows
    within the organization, even though the Owner (unlike a plain
    Member) can also see the org-wide totals in the same response.
    """

    def setUp(self):
        from .services.organization_service import create_organization

        self.org_type = OrganizationType.objects.create(slug="my_activity_org_type", name="Company")
        self.owner = User.objects.create_user(username="my_activity_owner", password="pw12345")
        self.organization = create_organization(name="My Activity Org", org_type=self.org_type, created_by=self.owner)

        self.other_member = User.objects.create_user(username="my_activity_member", password="pw12345")
        OrganizationMembership.objects.create(organization=self.organization, user=self.other_member, role=OrganizationMembership.Role.MEMBER)

        Document.objects.create(
            user=self.owner, title="Owner Doc", file="documents/owner.txt", file_hash="my_activity_fh1", organization=self.organization,
        )
        Document.objects.create(
            user=self.other_member, title="Member Doc", file="documents/member.txt", file_hash="my_activity_fh2", organization=self.organization,
        )

    def test_only_counts_the_requesting_users_own_documents(self):
        from .services.stats_service import get_my_activity_summary

        owner_activity = get_my_activity_summary(self.owner, self.organization)
        self.assertEqual(owner_activity["documents"], 1)

        member_activity = get_my_activity_summary(self.other_member, self.organization)
        self.assertEqual(member_activity["documents"], 1)

    def test_organization_stats_api_includes_my_activity_scoped_to_the_owner_only(self):
        # No platform-RBAC (Permission/Role/UserRole) setup needed here -
        # organization_stats_view is gated by HasOrgPermission
        # ("organization.view"), the org-scoped permission system, which
        # self.owner already satisfies automatically as this org's Owner
        # (create_organization() makes its creator the Owner) - a
        # completely separate system from platform RBAC (see
        # org_permission_service.py's module docstring).
        self.client.login(username="my_activity_owner", password="pw12345")
        response = self.client.get(f"/api/organizations/{self.organization.slug}/stats/")
        self.assertEqual(response.status_code, 200)

        my_activity = response.json()["my_activity"]
        self.assertEqual(my_activity["documents"], 1)
        # The whole-org figure in the same payload must still be 2 (both
        # the Owner's and the Member's document) - my_activity narrows
        # without narrowing the rest of the response.
        self.assertEqual(response.json()["document_count"], 2)


class OrgMemberRegistrationServiceTests(TestCase):
    """
    org_member_registration_service.py - the 2026-09-06 replacement
    for the email-invitation-link flow. An Owner registers a member by
    name + email; the service generates a username/password, creates
    (or reactivates) the User directly, and requires a password change
    on first login. No personal workspace ever exists for this account
    (account_type is COMPANY from the moment it's created).
    """

    def setUp(self):
        from .services.organization_service import create_organization

        self.org_type = OrganizationType.objects.create(slug="reg_svc_org_type", name="Company")
        self.owner = User.objects.create_user(username="reg_owner", password="pw12345")
        self.organization = create_organization(name="Registration Co", org_type=self.org_type, created_by=self.owner)

    def test_register_creates_an_active_company_only_account(self):
        from .services.org_member_registration_service import register_company_member

        membership = register_company_member(self.organization, "Ada Lovelace", "ada@example.com", self.owner)

        self.assertEqual(membership.role, OrganizationMembership.Role.MEMBER)
        user = membership.user
        self.assertTrue(user.is_active)
        self.assertTrue(user.profile.must_change_password)
        self.assertEqual(user.profile.account_type, UserProfile.AccountType.COMPANY)
        self.assertTrue(user.username)  # auto-generated, non-empty
        self.assertNotEqual(user.username, "")

    def test_register_sends_credentials_email(self):
        """
        Calls the email task directly rather than through
        register_company_member()'s real task_runner.submit()
        dispatch - same reasoning OrgInvitationServiceTests.
        test_send_org_invitation_email_task_delivers_real_email uses:
        the background thread pool worker opens its own DB connection,
        which can't see this plain TestCase's uncommitted outer
        transaction, so a User created moments ago in this same test
        would look like it "no longer exists" to that thread (a real,
        previously-observed failure mode here, not hypothetical).
        """
        from django.core import mail

        from .services.org_member_registration_service import register_company_member

        membership = register_company_member(self.organization, "Grace Hopper", "grace@example.com", self.owner)

        tasks.send_org_member_credentials_email_task(membership.user_id, self.organization.id, "some-generated-password")

        self.assertEqual(len(mail.outbox), 1)
        sent = mail.outbox[0]
        self.assertEqual(sent.to, ["grace@example.com"])
        self.assertIn(self.organization.name, sent.subject)

    def test_username_collision_gets_a_numeric_suffix(self):
        from .services.org_member_registration_service import register_company_member

        first = register_company_member(self.organization, "Ada Lovelace", "ada1@example.com", self.owner)
        second = register_company_member(self.organization, "Ada Lovelace", "ada2@example.com", self.owner)

        self.assertNotEqual(first.user.username, second.user.username)

    def test_registering_the_same_email_twice_in_one_org_is_rejected(self):
        from .services.org_member_registration_service import MemberRegistrationError, register_company_member

        register_company_member(self.organization, "Ada Lovelace", "dup@example.com", self.owner)
        with self.assertRaises(MemberRegistrationError):
            register_company_member(self.organization, "Ada Lovelace", "dup@example.com", self.owner)

    def test_registering_an_email_with_an_existing_active_account_is_rejected(self):
        from .services.org_member_registration_service import MemberRegistrationError, register_company_member

        User.objects.create_user(username="already_here", password="pw", email="taken@example.com")
        with self.assertRaises(MemberRegistrationError):
            register_company_member(self.organization, "Someone Else", "taken@example.com", self.owner)

    def test_removed_member_can_be_reactivated_by_re_registering(self):
        """The reversible-removal path org_membership_service.remove_member() promises: re-registering the same email reactivates the SAME account rather than creating a duplicate."""
        from .services.org_membership_service import remove_member
        from .services.org_member_registration_service import register_company_member

        first = register_company_member(self.organization, "Ada Lovelace", "ada@example.com", self.owner)
        original_user_id = first.user_id
        owner_membership = OrganizationMembership.objects.get(organization=self.organization, user=self.owner)

        remove_member(self.organization, owner_membership, first)
        first.user.refresh_from_db()
        self.assertFalse(first.user.is_active)

        second = register_company_member(self.organization, "Ada Lovelace", "ada@example.com", self.owner)
        self.assertEqual(second.user_id, original_user_id)
        self.assertTrue(second.user.is_active)

    def test_seat_limit_blocks_registration(self):
        from .services import billing_service

        low_seat_plan = Plan.objects.create(name="Tiny Reg", slug="tiny-reg", max_seats=1)
        billing_service.assign_plan(self.organization, low_seat_plan, self.owner)

        from .services.org_member_registration_service import register_company_member

        with self.assertRaises(billing_service.UsageLimitExceeded) as ctx:
            register_company_member(self.organization, "Ada Lovelace", "ada@example.com", self.owner)
        self.assertEqual(ctx.exception.limit_type, "seats")

    def test_api_view_rejects_a_plain_member(self):
        member = User.objects.create_user(username="reg_member", password="pw12345")
        OrganizationMembership.objects.create(organization=self.organization, user=member, role=OrganizationMembership.Role.MEMBER)

        self.client.login(username="reg_member", password="pw12345")
        response = self.client.post(
            f"/api/organizations/{self.organization.slug}/members/register/",
            {"full_name": "New Person", "email": "newperson@example.com"},
        )
        self.assertEqual(response.status_code, 403)

    def test_api_view_allows_the_owner(self):
        self.client.login(username="reg_owner", password="pw12345")
        response = self.client.post(
            f"/api/organizations/{self.organization.slug}/members/register/",
            {"full_name": "New Person", "email": "newperson@example.com"},
        )
        self.assertEqual(response.status_code, 201)
        self.assertTrue(OrganizationMembership.objects.filter(organization=self.organization, user__email="newperson@example.com").exists())


class ForcedPasswordChangeTests(TestCase):
    """
    UserProfile.must_change_password - set by member registration,
    surfaced in the session payload, cleared by a successful password
    change through the ordinary profile_password_view endpoint (no
    separate "first password" endpoint to keep in sync).
    """

    def setUp(self):
        from .services.organization_service import create_organization
        from .services.org_member_registration_service import register_company_member

        self.org_type = OrganizationType.objects.create(slug="forced_pw_org_type", name="Company")
        self.owner = User.objects.create_user(username="forced_pw_owner", password="pw12345")
        self.organization = create_organization(name="Forced PW Co", org_type=self.org_type, created_by=self.owner)

        membership = register_company_member(self.organization, "New Member", "member@example.com", self.owner)
        self.member_user = membership.user
        # The plaintext password only ever exists transiently inside
        # register_company_member()/the email task - recovered here by
        # regenerating a KNOWN password via set_password for test
        # purposes rather than trying to intercept the generated one.
        self.known_password = "KnownTempPass123!"
        self.member_user.set_password(self.known_password)
        self.member_user.save(update_fields=["password"])

    def test_session_reports_must_change_password(self):
        self.client.login(username=self.member_user.username, password=self.known_password)
        response = self.client.get("/api/auth/session/")
        self.assertTrue(response.json()["must_change_password"])

    def test_changing_password_clears_the_flag(self):
        self.client.login(username=self.member_user.username, password=self.known_password)
        response = self.client.post("/api/profile/password/", {
            "old_password": self.known_password,
            "new_password1": "BrandNewPass456!",
            "new_password2": "BrandNewPass456!",
        })
        self.assertEqual(response.status_code, 200)

        self.member_user.refresh_from_db()
        self.assertFalse(self.member_user.profile.must_change_password)

    def test_self_service_signup_never_sets_the_flag(self):
        personal = User.objects.create_user(username="normal_signup_user", password="pw12345")
        self.assertFalse(personal.profile.must_change_password)


class AICreditsTests(TestCase):
    """
    billing_service.py's AI credit ledger - the organization's SOLE
    spend-metering currency (2026-09-08 redesign; the Plan's own
    max_queries_per_month/max_ai_task_runs_per_month are informational
    display only now, see BillingTests above and billing_service.py's
    module docstring). Only an Owner can top up; every question asked
    and every AI Task run spends credits automatically.
    """

    def setUp(self):
        from .services.organization_service import create_organization

        self.org_type = OrganizationType.objects.create(slug="ai_credits_org_type", name="Company")
        self.owner = User.objects.create_user(username="credits_owner", password="pw12345")
        self.organization = create_organization(name="Credits Co", org_type=self.org_type, created_by=self.owner)

    def test_new_organization_has_unlimited_credits_until_an_owner_tops_up(self):
        """NULL (not 0) is the default - a brand-new organization must never be silently blocked from AI features it never asked to meter. See Organization.ai_credits_balance's help_text."""
        from .services import billing_service

        self.assertIsNone(self.organization.ai_credits_balance)
        billing_service.check_ai_credits(self.organization, 999999)  # must not raise
        billing_service.deduct_ai_credits(self.organization, 999999, "spend against unlimited", actor=self.owner)
        self.organization.refresh_from_db()
        self.assertIsNone(self.organization.ai_credits_balance)  # still untouched - nothing was ever activated

    def test_personal_workspace_is_never_subject_to_credits(self):
        from .services import billing_service
        billing_service.check_ai_credits(None, 999999)  # must not raise

    def test_add_credits_records_a_ledger_entry(self):
        from .services import billing_service
        from .models import AICreditTransaction

        balance = billing_service.add_ai_credits(self.organization, 100, self.owner)
        self.assertEqual(balance, 100)
        self.organization.refresh_from_db()
        self.assertEqual(self.organization.ai_credits_balance, 100)

        entry = AICreditTransaction.objects.get(organization=self.organization)
        self.assertEqual(entry.amount, 100)
        self.assertEqual(entry.balance_after, 100)
        self.assertEqual(entry.actor, self.owner)

    def test_add_credits_rejects_non_positive_amounts(self):
        from .services import billing_service

        with self.assertRaises(billing_service.BillingServiceError):
            billing_service.add_ai_credits(self.organization, 0, self.owner)
        with self.assertRaises(billing_service.BillingServiceError):
            billing_service.add_ai_credits(self.organization, -5, self.owner)

    def test_check_blocks_once_balance_is_insufficient(self):
        from .services import billing_service

        billing_service.add_ai_credits(self.organization, 10, self.owner)
        billing_service.check_ai_credits(self.organization, 10)  # exactly enough - must not raise

        with self.assertRaises(billing_service.UsageLimitExceeded) as ctx:
            billing_service.check_ai_credits(self.organization, 11)
        self.assertEqual(ctx.exception.limit_type, "ai_credits")

    def test_deduct_never_goes_negative(self):
        from .services import billing_service

        billing_service.add_ai_credits(self.organization, 5, self.owner)
        billing_service.deduct_ai_credits(self.organization, 999, "over-spend", actor=self.owner)
        self.organization.refresh_from_db()
        self.assertEqual(self.organization.ai_credits_balance, 0)

    def test_asking_a_question_spends_one_credit(self):
        from unittest.mock import patch

        from .services import billing_service

        billing_service.add_ai_credits(self.organization, 10, self.owner)

        permission, _ = Permission.objects.get_or_create(codename="pages.ask_ai", defaults={"name": "pages.ask_ai"})
        role = Role.objects.create(slug="ai_credits_ask_role", name="User")
        role.permissions.add(permission)
        UserRole.objects.create(user=self.owner, role=role)

        with patch("RAG.api.ask_views.answer_question", return_value={"answer": "hi", "sources": [], "confidence": 50}):
            self.client.login(username="credits_owner", password="pw12345")
            response = self.client.post(
                "/api/ask/", data=json.dumps({"question": "hello"}), content_type="application/json",
                HTTP_X_ORGANIZATION_SLUG=self.organization.slug,
            )
        self.assertEqual(response.status_code, 200)
        self.organization.refresh_from_db()
        self.assertEqual(self.organization.ai_credits_balance, 9)

    def test_asking_a_question_is_blocked_at_zero_credits(self):
        from unittest.mock import patch

        permission, _ = Permission.objects.get_or_create(codename="pages.ask_ai", defaults={"name": "pages.ask_ai"})
        role = Role.objects.create(slug="ai_credits_ask_role_2", name="User")
        role.permissions.add(permission)
        UserRole.objects.create(user=self.owner, role=role)

        with patch("RAG.api.ask_views.answer_question") as mock_answer:
            self.client.login(username="credits_owner", password="pw12345")
            response = self.client.post(
                "/api/ask/", data=json.dumps({"question": "hello"}), content_type="application/json",
                HTTP_X_ORGANIZATION_SLUG=self.organization.slug,
            )
        self.assertEqual(response.status_code, 402)
        self.assertEqual(response.json()["code"], "usage_limit_exceeded")
        mock_answer.assert_not_called()

    def test_member_cannot_view_or_manage_ai_credits(self):
        member = User.objects.create_user(username="credits_member", password="pw12345")
        OrganizationMembership.objects.create(organization=self.organization, user=member, role=OrganizationMembership.Role.MEMBER)

        self.client.login(username="credits_member", password="pw12345")
        response = self.client.get(f"/api/organizations/{self.organization.slug}/ai-credits/")
        self.assertEqual(response.status_code, 403)

    def test_owner_can_view_and_top_up_credits_via_api(self):
        self.client.login(username="credits_owner", password="pw12345")

        get_response = self.client.get(f"/api/organizations/{self.organization.slug}/ai-credits/")
        self.assertEqual(get_response.status_code, 200)
        self.assertEqual(get_response.json()["balance"], 0)

        post_response = self.client.post(f"/api/organizations/{self.organization.slug}/ai-credits/", {"amount": 250})
        self.assertEqual(post_response.status_code, 200)
        self.assertEqual(post_response.json()["balance"], 250)
        self.assertEqual(len(post_response.json()["transactions"]), 1)


class OrganizationQueriesViewTests(TestCase):
    """
    organization_queries_view - an Owner can see that questions were
    asked (who/when/confidence/method) but the API response never
    includes the actual question/answer text at all.
    """

    def setUp(self):
        from .services.organization_service import create_organization

        self.org_type = OrganizationType.objects.create(slug="org_queries_org_type", name="Company")
        self.owner = User.objects.create_user(username="queries_owner", password="pw12345")
        self.organization = create_organization(name="Queries Co", org_type=self.org_type, created_by=self.owner)

        self.member = User.objects.create_user(username="queries_member", password="pw12345")
        OrganizationMembership.objects.create(organization=self.organization, user=self.member, role=OrganizationMembership.Role.MEMBER)

        QueryLog.objects.create(
            user=self.member, organization=self.organization,
            question="What is our secret roadmap?", answer="The secret answer.",
            search_method="Hybrid (Vector + BM25)", response_time_ms=42, confidence=88,
        )

    def test_member_cannot_view_the_queries_log(self):
        self.client.login(username="queries_member", password="pw12345")
        response = self.client.get(f"/api/organizations/{self.organization.slug}/queries/")
        self.assertEqual(response.status_code, 403)

    def test_owner_sees_metadata_but_never_question_or_answer_text(self):
        self.client.login(username="queries_owner", password="pw12345")
        response = self.client.get(f"/api/organizations/{self.organization.slug}/queries/")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["count"], 1)
        entry = data["queries"][0]

        self.assertEqual(entry["user"], "queries_member")
        self.assertEqual(entry["confidence"], 88)
        self.assertEqual(entry["response_time_ms"], 42)
        self.assertNotIn("question", entry)
        self.assertNotIn("answer", entry)

        # Belt and braces: the sensitive strings must not appear ANYWHERE
        # in the raw response body, not just absent as named fields.
        raw_body = response.content.decode()
        self.assertNotIn("secret roadmap", raw_body)
        self.assertNotIn("secret answer", raw_body)


class PlanFeatureGatingTests(TestCase):
    """
    Plan.included_features - the platform-admin -> company feature
    gate. Covers: empty included_features means unrestricted (every
    Plan created before this field existed keeps granting full
    access), a Plan exclusion blocks access even with no member
    override, and the exclusion applies uniformly (an Owner is not
    exempt from their own company's Plan - only from per-member
    overrides, a separate mechanism covered in
    MemberFeatureAccessTests).
    """

    def setUp(self):
        from .services import billing_service
        from .services.organization_service import create_organization

        self.billing_service = billing_service

        self.org_type = OrganizationType.objects.create(slug="plan_feature_org_type", name="Company")
        self.owner = User.objects.create_user(username="plan_feature_owner", password="pw12345")
        self.organization = create_organization(name="Plan Feature Org", org_type=self.org_type, created_by=self.owner)

        permissions = ["pages.ai_tasks", "pages.analytics", "pages.reports", "pages.knowledge_base"]
        self.role = Role.objects.create(slug="plan_feature_user", name="User")
        for codename in permissions:
            permission, _ = Permission.objects.get_or_create(codename=codename, defaults={"name": codename})
            self.role.permissions.add(permission)
        UserRole.objects.create(user=self.owner, role=self.role)

    def test_no_subscription_is_unrestricted(self):
        self.assertIsNone(self.billing_service.org_plan_feature_codes(self.organization))
        self.assertTrue(self.billing_service.org_has_plan_feature(self.organization, "reports"))

    def test_empty_included_features_is_unrestricted(self):
        plan = Plan.objects.create(name="Everything", slug="pfg-everything", included_features=[])
        self.billing_service.assign_plan(self.organization, plan, self.owner)
        self.assertIsNone(self.billing_service.org_plan_feature_codes(self.organization))
        self.assertTrue(self.billing_service.org_has_plan_feature(self.organization, "ai_tasks"))

    def test_plan_exclusion_blocks_even_with_no_member_override(self):
        plan = Plan.objects.create(name="Limited", slug="pfg-limited", included_features=["ai_tasks", "analytics"])
        self.billing_service.assign_plan(self.organization, plan, self.owner)

        self.assertTrue(self.billing_service.org_has_plan_feature(self.organization, "ai_tasks"))
        self.assertFalse(self.billing_service.org_has_plan_feature(self.organization, "reports"))

    def test_plan_exclusion_applies_even_to_the_owner(self):
        from .services.org_member_feature_service import has_feature_access

        plan = Plan.objects.create(name="No Reports", slug="pfg-no-reports", included_features=["ai_tasks"])
        self.billing_service.assign_plan(self.organization, plan, self.owner)

        self.assertFalse(has_feature_access(self.owner, self.organization, "reports"))
        self.assertTrue(has_feature_access(self.owner, self.organization, "ai_tasks"))

    def test_reports_endpoint_403s_when_plan_excludes_it(self):
        plan = Plan.objects.create(name="No Reports HTTP", slug="pfg-no-reports-http", included_features=["ai_tasks"])
        self.billing_service.assign_plan(self.organization, plan, self.owner)

        self.client.login(username="plan_feature_owner", password="pw12345")
        response = self.client.get("/api/reports/", HTTP_X_ORGANIZATION_SLUG=self.organization.slug)
        self.assertEqual(response.status_code, 403)

    def test_create_plan_validates_unknown_feature_codes(self):
        with self.assertRaises(self.billing_service.BillingServiceError):
            self.billing_service.create_plan(name="Bad Plan", included_features=["not_a_real_feature"])


class MemberFeatureAccessTests(TestCase):
    """
    OrganizationMembership.disabled_features - Owner-controlled
    per-member restriction, bounded by the company's Plan. Covers: an
    Owner can restrict one Member without affecting another, an Owner
    cannot restrict another Owner (or themselves), unknown feature
    codes are rejected, and a member override can only narrow access -
    never re-enable a Plan-excluded feature.
    """

    def setUp(self):
        from .services import billing_service
        from .services import org_permission_service as org_perm
        from .services.organization_service import create_organization

        self.billing_service = billing_service
        self.org_perm = org_perm

        self.org_type = OrganizationType.objects.create(slug="member_feature_org_type", name="Company")
        self.owner = User.objects.create_user(username="member_feature_owner", password="pw12345")
        self.organization = create_organization(name="Member Feature Org", org_type=self.org_type, created_by=self.owner)

        self.member_a = User.objects.create_user(username="member_feature_a", password="pw12345")
        self.member_b = User.objects.create_user(username="member_feature_b", password="pw12345")
        self.membership_a = OrganizationMembership.objects.create(organization=self.organization, user=self.member_a, role=org_perm.MEMBER)
        self.membership_b = OrganizationMembership.objects.create(organization=self.organization, user=self.member_b, role=org_perm.MEMBER)
        self.owner_membership = OrganizationMembership.objects.get(organization=self.organization, user=self.owner)

        permissions = ["pages.ai_tasks", "pages.analytics", "pages.reports", "pages.knowledge_base"]
        self.role = Role.objects.create(slug="member_feature_user", name="User")
        for codename in permissions:
            permission, _ = Permission.objects.get_or_create(codename=codename, defaults={"name": codename})
            self.role.permissions.add(permission)
        for user in (self.owner, self.member_a, self.member_b):
            UserRole.objects.create(user=user, role=self.role)

    def test_owner_can_restrict_one_member_without_affecting_another(self):
        from .services.org_member_feature_service import has_feature_access, set_member_disabled_features

        set_member_disabled_features(self.organization, self.owner_membership, self.membership_a, ["analytics"])

        self.assertFalse(has_feature_access(self.member_a, self.organization, "analytics"))
        self.assertTrue(has_feature_access(self.member_b, self.organization, "analytics"))
        # Untouched features for the restricted member are unaffected.
        self.assertTrue(has_feature_access(self.member_a, self.organization, "ai_tasks"))

    def test_owner_cannot_restrict_another_owner(self):
        from .services.org_member_feature_service import FeatureAccessError, set_member_disabled_features

        second_owner = User.objects.create_user(username="member_feature_second_owner", password="pw12345")
        second_owner_membership = OrganizationMembership.objects.create(
            organization=self.organization, user=second_owner, role=self.org_perm.OWNER,
        )

        with self.assertRaises(FeatureAccessError):
            set_member_disabled_features(self.organization, self.owner_membership, second_owner_membership, ["analytics"])

    def test_unknown_feature_code_is_rejected(self):
        from .services.org_member_feature_service import FeatureAccessError, set_member_disabled_features

        with self.assertRaises(FeatureAccessError):
            set_member_disabled_features(self.organization, self.owner_membership, self.membership_a, ["not_a_real_feature"])

    def test_member_override_cannot_exceed_the_plan(self):
        from .services.org_member_feature_service import has_feature_access, set_member_disabled_features

        plan = Plan.objects.create(name="No Reports Member", slug="mfa-no-reports", included_features=["ai_tasks", "analytics"])
        self.billing_service.assign_plan(self.organization, plan, self.owner)

        # member_a has no override at all for "reports" - the Plan alone excludes it.
        self.assertFalse(has_feature_access(self.member_a, self.organization, "reports"))

        # Explicitly "clearing" any restriction (empty disabled_features) still
        # can't grant a Plan-excluded feature - there is no code path that widens
        # beyond the Plan.
        set_member_disabled_features(self.organization, self.owner_membership, self.membership_a, [])
        self.assertFalse(has_feature_access(self.member_a, self.organization, "reports"))
        self.assertTrue(has_feature_access(self.member_a, self.organization, "analytics"))

    def test_feature_gated_endpoint_403s_for_the_restricted_member_only(self):
        """
        Uses AI Tasks config, not Analytics - company Analytics/Reports
        are Owner-only by product decision now (user_can_view_org_analytics,
        see analytics_views.py's docstring), independent of per-member
        disabled_features, so a plain Member 403s there regardless of
        this override and can't tell the two mechanisms apart. AI Tasks
        has no such Owner-only gate, so it's the one already-member-
        accessible, feature-gated endpoint that actually isolates what
        this test means to check: one member's override doesn't leak
        onto another's.
        """
        from .services.org_member_feature_service import set_member_disabled_features

        set_member_disabled_features(self.organization, self.owner_membership, self.membership_a, ["ai_tasks"])

        self.client.login(username="member_feature_a", password="pw12345")
        restricted_response = self.client.get("/api/ai-tasks/config/", HTTP_X_ORGANIZATION_SLUG=self.organization.slug)
        self.assertEqual(restricted_response.status_code, 403)

        self.client.login(username="member_feature_b", password="pw12345")
        unrestricted_response = self.client.get("/api/ai-tasks/config/", HTTP_X_ORGANIZATION_SLUG=self.organization.slug)
        self.assertEqual(unrestricted_response.status_code, 200)

    def test_stale_disabled_features_never_restrict_an_owner(self):
        """
        Regression: a real company ("Riven") had its Owner's own
        OrganizationMembership.disabled_features populated with
        ["analytics", "reports"] (leftover from before they were
        promoted from Member to Owner via update_member_role(), which
        didn't clear it) and has_feature_access() enforced it, silently
        locking the Owner out of Analytics/Reports despite the org's
        Plan including both. set_member_disabled_features() already
        refuses to ever WRITE this for an Owner - this covers the read
        side matching that same "an Owner is never restricted"
        contract regardless of how a non-empty list ended up on their
        row.
        """
        from .services.org_member_feature_service import has_feature_access

        self.owner_membership.disabled_features = ["analytics", "reports"]
        self.owner_membership.save(update_fields=["disabled_features"])

        self.assertTrue(has_feature_access(self.owner, self.organization, "analytics"))
        self.assertTrue(has_feature_access(self.owner, self.organization, "reports"))

    def test_promoting_a_member_to_owner_clears_their_disabled_features(self):
        from .services.org_member_feature_service import set_member_disabled_features
        from .services.org_membership_service import update_member_role

        set_member_disabled_features(self.organization, self.owner_membership, self.membership_a, ["analytics", "reports"])
        self.assertEqual(self.membership_a.disabled_features, ["analytics", "reports"])

        update_member_role(self.organization, self.owner_membership, self.membership_a, self.org_perm.OWNER)
        self.membership_a.refresh_from_db()
        self.assertEqual(self.membership_a.disabled_features, [])


class MemberGrantedAnalyticsReportsAccessTests(TestCase):
    """
    Regression: a real company Owner ("Riven") reported being unable to
    grant a Member access to Analytics/Reports even though the org's
    Plan includes both - user_can_view_org_analytics() categorically
    blocked every non-Owner, so the per-member Feature Access toggle
    for "analytics"/"reports" (shown in the UI, backed by
    has_feature_access()) was a no-op for those two specific features.
    Covers: a Member has access to both by default (same as every
    other FEATURE_CODES entry - disabled_features starts empty), the
    Owner can still individually restrict a Member, and a Member let in
    sees only their own activity, never the whole company's.
    """

    def setUp(self):
        from .services.organization_service import create_organization
        from .services.org_permission_service import MEMBER

        self.org_type = OrganizationType.objects.create(slug="member_analytics_org_type", name="Company")
        self.owner = User.objects.create_user(username="member_analytics_owner", password="pw12345")
        self.organization = create_organization(name="Member Analytics Co", org_type=self.org_type, created_by=self.owner)

        self.member_a = User.objects.create_user(username="member_analytics_a", password="pw12345")
        self.member_b = User.objects.create_user(username="member_analytics_b", password="pw12345")
        self.membership_a = OrganizationMembership.objects.create(organization=self.organization, user=self.member_a, role=MEMBER)
        OrganizationMembership.objects.create(organization=self.organization, user=self.member_b, role=MEMBER)
        self.owner_membership = OrganizationMembership.objects.get(organization=self.organization, user=self.owner)

        role = Role.objects.create(slug="member_analytics_default_user", name="User")
        for codename in ("pages.analytics", "pages.reports"):
            permission, _ = Permission.objects.get_or_create(codename=codename, defaults={"name": codename})
            role.permissions.add(permission)
        for user in (self.owner, self.member_a, self.member_b):
            UserRole.objects.create(user=user, role=role)

        Document.objects.create(
            user=self.member_a, organization=self.organization, title="Member A's Doc.pdf",
            file="documents/member-a-test.pdf", file_type="pdf", file_size=1024, file_hash="member-a-hash",
        )
        Document.objects.create(
            user=self.member_b, organization=self.organization, title="Member B's Doc.pdf",
            file="documents/member-b-test.pdf", file_type="pdf", file_size=2048, file_hash="member-b-hash",
        )

    def test_member_has_analytics_and_reports_access_by_default(self):
        self.client.login(username="member_analytics_a", password="pw12345")
        analytics_response = self.client.get("/api/analytics/", HTTP_X_ORGANIZATION_SLUG=self.organization.slug)
        self.assertEqual(analytics_response.status_code, 200)
        reports_response = self.client.get("/api/reports/", HTTP_X_ORGANIZATION_SLUG=self.organization.slug)
        self.assertEqual(reports_response.status_code, 200)

    def test_owner_can_still_restrict_a_specific_members_analytics_and_reports(self):
        from .services.org_member_feature_service import set_member_disabled_features

        set_member_disabled_features(self.organization, self.owner_membership, self.membership_a, ["analytics", "reports"])

        self.client.login(username="member_analytics_a", password="pw12345")
        self.assertEqual(self.client.get("/api/analytics/", HTTP_X_ORGANIZATION_SLUG=self.organization.slug).status_code, 403)
        self.assertEqual(self.client.get("/api/reports/", HTTP_X_ORGANIZATION_SLUG=self.organization.slug).status_code, 403)

        # member_b was never restricted - unaffected.
        self.client.login(username="member_analytics_b", password="pw12345")
        self.assertEqual(self.client.get("/api/analytics/", HTTP_X_ORGANIZATION_SLUG=self.organization.slug).status_code, 200)

    def test_member_reports_are_scoped_to_their_own_documents_only(self):
        self.client.login(username="member_analytics_a", password="pw12345")
        response = self.client.get("/api/reports/", HTTP_X_ORGANIZATION_SLUG=self.organization.slug)
        self.assertEqual(response.status_code, 200)
        # Sees only their own upload, never member_b's - never the
        # whole-company total the Owner would see (2).
        self.assertEqual(response.json()["document_count"], 1)

    def test_owner_reports_still_see_the_whole_companys_documents(self):
        self.client.login(username="member_analytics_owner", password="pw12345")
        response = self.client.get("/api/reports/", HTTP_X_ORGANIZATION_SLUG=self.organization.slug)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["document_count"], 2)


class OwnerOrgLibraryAccessTests(TestCase):
    """
    can_manage_org_library() (RAG.services.document_access_service) -
    "documents.manage_org_library" is NOT in seed_rbac.py's
    USER_DEFAULT_PERMISSIONS, so ordinarily only an Admin-granted
    custom role holds it. This covers the fix letting an organization's
    own Owner curate their company's Organization Library without that
    separately-granted platform permission - the same org-management
    rank they already hold everywhere else (members, billing,
    settings) - while a plain Member (no special role) still cannot.
    """

    def setUp(self):
        from .services.organization_service import create_organization
        from .services.org_permission_service import MEMBER, OWNER

        self.org_type = OrganizationType.objects.create(slug="org_library_org_type", name="Company")
        self.owner = User.objects.create_user(username="org_library_owner", password="pw12345")
        self.organization = create_organization(name="Org Library Co", org_type=self.org_type, created_by=self.owner)

        self.member = User.objects.create_user(username="org_library_member", password="pw12345")
        OrganizationMembership.objects.create(organization=self.organization, user=self.member, role=MEMBER)

        # USER_DEFAULT_PERMISSIONS-equivalent - "pages.documents" only,
        # deliberately NOT "documents.manage_org_library" - so this
        # exercises the Owner-rank bypass, not a granted permission.
        role = Role.objects.create(slug="org_library_default_user", name="User")
        permission, _ = Permission.objects.get_or_create(codename="pages.documents", defaults={"name": "pages.documents"})
        role.permissions.add(permission)
        UserRole.objects.create(user=self.owner, role=role)
        UserRole.objects.create(user=self.member, role=role)

        self.document = Document.objects.create(
            user=self.member, organization=self.organization, title="Team Handbook.pdf",
            file="documents/org-library-test.pdf", file_type="pdf", file_size=1024, file_hash="org-library-test-hash",
        )

    def test_owner_can_toggle_a_document_into_the_org_library_without_the_platform_permission(self):
        self.client.login(username="org_library_owner", password="pw12345")
        response = self.client.post(
            f"/api/documents/org-library/{self.document.id}/toggle/", HTTP_X_ORGANIZATION_SLUG=self.organization.slug,
        )
        self.assertEqual(response.status_code, 200)
        self.document.refresh_from_db()
        self.assertTrue(self.document.is_org_library)

        # Toggling again removes it - same endpoint, same Owner.
        response = self.client.post(
            f"/api/documents/org-library/{self.document.id}/toggle/", HTTP_X_ORGANIZATION_SLUG=self.organization.slug,
        )
        self.assertEqual(response.status_code, 200)
        self.document.refresh_from_db()
        self.assertFalse(self.document.is_org_library)

    def test_plain_member_cannot_toggle_the_org_library(self):
        self.client.login(username="org_library_member", password="pw12345")
        response = self.client.post(
            f"/api/documents/org-library/{self.document.id}/toggle/", HTTP_X_ORGANIZATION_SLUG=self.organization.slug,
        )
        self.assertEqual(response.status_code, 403)
        self.document.refresh_from_db()
        self.assertFalse(self.document.is_org_library)

    def test_org_library_view_reports_can_manage_correctly_for_owner_and_member(self):
        self.client.login(username="org_library_owner", password="pw12345")
        owner_response = self.client.get("/api/documents/org-library/", HTTP_X_ORGANIZATION_SLUG=self.organization.slug)
        self.assertEqual(owner_response.status_code, 200)
        self.assertTrue(owner_response.json()["can_manage"])

        self.client.login(username="org_library_member", password="pw12345")
        member_response = self.client.get("/api/documents/org-library/", HTTP_X_ORGANIZATION_SLUG=self.organization.slug)
        self.assertEqual(member_response.status_code, 200)
        self.assertFalse(member_response.json()["can_manage"])


class PlanChangeRequestTests(TestCase):
    """
    billing_service.request_plan_change()/approve_plan_request()/
    reject_plan_request() - the Owner/Personal-user self-service
    request queue in front of assign_plan()/assign_personal_plan().
    Approval is the only thing that actually writes a Subscription;
    rejection leaves the org/user exactly as it was.
    """

    def setUp(self):
        from .services.organization_service import create_organization
        from .services.billing_service import create_plan

        self.org_type = OrganizationType.objects.create(slug="plan_req_org_type", name="Company")
        self.owner = User.objects.create_user(username="plan_req_owner", password="pw12345")
        self.organization = create_organization(name="Plan Req Org", org_type=self.org_type, created_by=self.owner)

        self.admin = User.objects.create_user(username="plan_req_admin", password="pw12345", is_staff=True)
        permission = Permission.objects.create(codename="billing.manage_plans", name="Manage Billing Plans")
        role = Role.objects.create(slug="plan_req_admin_role", name="Billing Admin")
        role.permissions.add(permission)
        UserRole.objects.create(user=self.admin, role=role)

        self.plan_a = create_plan(name="Plan Req Starter", included_credits=100, actor=self.admin)
        self.plan_b = create_plan(name="Plan Req Pro", included_credits=500, actor=self.admin)

    def test_request_then_approve_activates_the_plan_and_grants_credits(self):
        from .services.billing_service import approve_plan_request, get_active_subscription, request_plan_change

        plan_request = request_plan_change(self.plan_a, self.owner, organization=self.organization)
        self.assertEqual(plan_request.status, "pending")
        self.assertIsNone(get_active_subscription(self.organization))

        approve_plan_request(plan_request, self.admin)
        plan_request.refresh_from_db()
        self.assertEqual(plan_request.status, "approved")
        self.assertEqual(plan_request.reviewed_by, self.admin)

        subscription = get_active_subscription(self.organization)
        self.assertEqual(subscription.plan, self.plan_a)
        self.organization.refresh_from_db()
        self.assertEqual(self.organization.ai_credits_balance, 100)

    def test_second_request_while_one_is_pending_is_rejected(self):
        from .services.billing_service import BillingServiceError, request_plan_change

        request_plan_change(self.plan_a, self.owner, organization=self.organization)
        with self.assertRaises(BillingServiceError):
            request_plan_change(self.plan_b, self.owner, organization=self.organization)

    def test_reject_leaves_the_organization_without_a_plan(self):
        from .services.billing_service import get_active_subscription, reject_plan_request, request_plan_change

        plan_request = request_plan_change(self.plan_a, self.owner, organization=self.organization)
        reject_plan_request(plan_request, self.admin, note="Not eligible yet")
        plan_request.refresh_from_db()
        self.assertEqual(plan_request.status, "rejected")
        self.assertEqual(plan_request.admin_note, "Not eligible yet")
        self.assertIsNone(get_active_subscription(self.organization))

    def test_approving_an_already_reviewed_request_is_rejected(self):
        from .services.billing_service import BillingServiceError, approve_plan_request, reject_plan_request, request_plan_change

        plan_request = request_plan_change(self.plan_a, self.owner, organization=self.organization)
        reject_plan_request(plan_request, self.admin)
        with self.assertRaises(BillingServiceError):
            approve_plan_request(plan_request, self.admin)

    def test_only_billing_manage_plans_holders_can_approve_via_the_api(self):
        from .services.billing_service import request_plan_change

        plan_request = request_plan_change(self.plan_a, self.owner, organization=self.organization)

        self.client.login(username="plan_req_owner", password="pw12345")
        response = self.client.post(f"/api/admin/billing/plan-requests/{plan_request.id}/action/", {"action": "approve"})
        self.assertEqual(response.status_code, 403)

        self.client.login(username="plan_req_admin", password="pw12345")
        response = self.client.post(f"/api/admin/billing/plan-requests/{plan_request.id}/action/", {"action": "approve"})
        self.assertEqual(response.status_code, 200)

    def test_company_plan_cannot_be_requested_for_a_personal_workspace(self):
        from .services.billing_service import BillingServiceError, request_plan_change

        personal_user = User.objects.create_user(username="plan_req_personal", password="pw12345")
        with self.assertRaises(BillingServiceError):
            request_plan_change(self.plan_a, personal_user, user=personal_user)


class CreditRefillTests(TestCase):
    """
    billing_service._refill_credits_for_subscription()/_advance_period()
    - credits reset to EXACTLY plan.included_credits (not additive) on
    genuine period rollover, and immediately on first assignment.
    """

    def setUp(self):
        from .services.organization_service import create_organization
        from .services.billing_service import create_plan

        self.org_type = OrganizationType.objects.create(slug="credit_refill_org_type", name="Company")
        self.owner = User.objects.create_user(username="credit_refill_owner", password="pw12345")
        self.organization = create_organization(name="Credit Refill Org", org_type=self.org_type, created_by=self.owner)
        self.plan = create_plan(name="Credit Refill Plan", included_credits=200, actor=self.owner)

    def test_first_assignment_immediately_grants_included_credits(self):
        from .services.billing_service import assign_plan

        assign_plan(self.organization, self.plan, self.owner)
        self.organization.refresh_from_db()
        self.assertEqual(self.organization.ai_credits_balance, 200)

    def test_period_rollover_resets_balance_to_included_credits_not_additive(self):
        from datetime import timedelta

        from django.utils import timezone

        from .services.billing_service import _advance_period, assign_plan, deduct_ai_credits

        subscription = assign_plan(self.organization, self.plan, self.owner)
        deduct_ai_credits(self.organization, 50, "Question asked", actor=self.owner)
        self.organization.refresh_from_db()
        self.assertEqual(self.organization.ai_credits_balance, 150)

        # Force the period to already be in the past so the next read rolls it over.
        subscription.current_period_end = timezone.now() - timedelta(days=1)
        subscription.save(update_fields=["current_period_end"])

        _advance_period(subscription)
        self.organization.refresh_from_db()
        self.assertEqual(self.organization.ai_credits_balance, 200)

    def test_mid_period_read_does_not_touch_the_balance(self):
        from .services.billing_service import _advance_period, assign_plan, deduct_ai_credits

        assign_plan(self.organization, self.plan, self.owner)
        deduct_ai_credits(self.organization, 30, "Question asked", actor=self.owner)
        subscription = self.organization.subscription

        _advance_period(subscription)  # period hasn't lapsed - should be a no-op
        self.organization.refresh_from_db()
        self.assertEqual(self.organization.ai_credits_balance, 170)

    def test_ledger_records_the_refill(self):
        from datetime import timedelta

        from django.utils import timezone

        from .services.billing_service import _advance_period, assign_plan, deduct_ai_credits

        subscription = assign_plan(self.organization, self.plan, self.owner)
        deduct_ai_credits(self.organization, 80, "Question asked", actor=self.owner)
        subscription.current_period_end = timezone.now() - timedelta(days=1)
        subscription.save(update_fields=["current_period_end"])

        _advance_period(subscription)
        refill_entry = self.organization.ai_credit_transactions.filter(reason="Plan period refill").first()
        self.assertIsNotNone(refill_entry)
        self.assertEqual(refill_entry.amount, 80)  # 200 - 120 remaining = +80 back to 200
        self.assertEqual(refill_entry.balance_after, 200)


class PersonalPlanFeatureGatingTests(TestCase):
    """
    org_member_feature_service.has_feature_access() for organization=None
    (Personal Workspace) - now goes through that user's own Personal
    Plan ceiling instead of always returning True.
    """

    def setUp(self):
        from .services.billing_service import create_plan

        self.user = User.objects.create_user(username="personal_feature_user", password="pw12345")
        self.plan = create_plan(
            name="Personal Feature Plan", plan_type="personal", included_features=["documents", "ask_ai"], actor=self.user,
        )

    def test_no_personal_plan_means_unrestricted(self):
        from .services.org_member_feature_service import has_feature_access

        self.assertTrue(has_feature_access(self.user, None, "analytics"))

    def test_personal_plan_restricts_to_its_included_features(self):
        from .services.billing_service import assign_personal_plan
        from .services.org_member_feature_service import has_feature_access

        assign_personal_plan(self.user, self.plan, self.user)
        self.assertTrue(has_feature_access(self.user, None, "ask_ai"))
        self.assertFalse(has_feature_access(self.user, None, "analytics"))


class AIRequestTraceOrganizationTests(TestCase):
    """
    AIRequestTrace.organization - denormalized at write time by
    observability_service.save_trace() for per-company/system-wide AI
    usage aggregation. Covers: organization is tagged correctly on
    save, NULL (Personal Workspace) is preserved, and the four newly
    HasOrgFeatureAccess-gated view files (knowledge, analytics,
    reports, ai_tasks) all 403 correctly under a Plan exclusion without
    affecting Personal Workspace access.
    """

    def setUp(self):
        from .services.organization_service import create_organization

        self.org_type = OrganizationType.objects.create(slug="trace_org_type", name="Company")
        self.user = User.objects.create_user(username="trace_org_user", password="pw12345")
        self.organization = create_organization(name="Trace Org", org_type=self.org_type, created_by=self.user)

        # create_organization() force-sets self.user to account_type=COMPANY
        # (see its docstring) - a header-less request from self.user
        # defaults to self.organization, NOT Personal Workspace, per
        # resolve_request_organization()'s account-type-driven default. A
        # genuinely separate, org-less account is needed to test real
        # Personal Workspace behavior below.
        self.personal_user = User.objects.create_user(username="trace_personal_user", password="pw12345")

        permissions = ["pages.ai_tasks", "pages.analytics", "pages.reports", "pages.knowledge_base"]
        self.role = Role.objects.create(slug="trace_org_role", name="User")
        for codename in permissions:
            permission, _ = Permission.objects.get_or_create(codename=codename, defaults={"name": codename})
            self.role.permissions.add(permission)
        for user in (self.user, self.personal_user):
            UserRole.objects.create(user=user, role=self.role)

    def test_save_trace_tags_the_given_organization(self):
        from .models import AIRequestTrace
        from .services.observability_service import save_trace

        trace = save_trace(
            "trace-org-test-1", AIRequestTrace.Source.ASK_AI, self.user,
            organization=self.organization, status=AIRequestTrace.Status.COMPLETED,
        )
        trace.refresh_from_db()
        self.assertEqual(trace.organization_id, self.organization.id)

    def test_save_trace_with_no_organization_stays_null(self):
        from .models import AIRequestTrace
        from .services.observability_service import save_trace

        trace = save_trace(
            "trace-org-test-2", AIRequestTrace.Source.ASK_AI, self.user,
            status=AIRequestTrace.Status.COMPLETED,
        )
        trace.refresh_from_db()
        self.assertIsNone(trace.organization_id)

    def test_all_four_gated_endpoints_403_under_plan_exclusion(self):
        from .services import billing_service

        plan = Plan.objects.create(name="Nothing", slug="trace-nothing-plan", included_features=[])
        billing_service.assign_plan(self.organization, plan, self.user)
        # Re-fetch with an actual exclusion - included_features=[] alone is
        # unrestricted, so this must explicitly name a restricted set.
        billing_service.update_plan(plan, included_features=["documents"])

        self.client.login(username="trace_org_user", password="pw12345")
        headers = {"HTTP_X_ORGANIZATION_SLUG": self.organization.slug}

        self.assertEqual(self.client.get("/api/knowledge/browse/", **headers).status_code, 403)
        self.assertEqual(self.client.get("/api/analytics/", **headers).status_code, 403)
        self.assertEqual(self.client.get("/api/reports/", **headers).status_code, 403)
        self.assertEqual(self.client.get("/api/ai-tasks/config/", **headers).status_code, 403)

    def test_personal_workspace_unaffected_by_any_plan_state(self):
        from .services import billing_service

        plan = Plan.objects.create(name="Nothing Personal", slug="trace-nothing-personal-plan", included_features=["documents"])
        billing_service.assign_plan(self.organization, plan, self.user)

        # trace_personal_user has no organization at all - a header-less
        # request from them is genuine Personal Workspace, which must
        # never be affected by ANY other organization's Plan state.
        self.client.login(username="trace_personal_user", password="pw12345")
        self.assertEqual(self.client.get("/api/analytics/").status_code, 200)
        self.assertEqual(self.client.get("/api/reports/").status_code, 200)


class AdminQueriesListPrivacyTests(TestCase):
    """
    Admin > Queries (/api/admin/queries/, and its CSV export) is
    deliberately metadata-only for EVERY viewer, full stop - not a
    "queries.view_content" toggle. Originally this was a regression
    test for a real bug (the list endpoint's serializer leaked raw
    `question` text to any "queries.view_all_logs" holder regardless
    of "queries.view_content", while the React table only ever
    *displayed* "Protected" for them - a cosmetic mask, not real
    enforcement); the product decision since then went further and
    removed the content-viewing feature entirely (no detail endpoint,
    no "with content" CSV variant, no content-text search) rather than
    leaving it as a permission a role could still be granted - see
    RAG.api.admin_queries_views' module docstring. So this now checks
    that NO combination of permissions ever surfaces question/answer
    text through this endpoint, not just that a content-less viewer is
    blocked.
    """

    def setUp(self):
        self.view_all_logs_permission, _ = Permission.objects.get_or_create(
            codename="queries.view_all_logs", defaults={"name": "View Query Logs"},
        )
        # Still exists in RBAC (Django's own /django-admin/ QueryLog
        # page uses it) - granting it here proves it no longer does
        # anything for THIS surface.
        self.content_permission, _ = Permission.objects.get_or_create(
            codename="queries.view_content", defaults={"name": "View Query Content"},
        )

        self.role = Role.objects.create(slug="queries_all_permissions", name="Every Queries Permission")
        self.role.permissions.add(self.view_all_logs_permission, self.content_permission)

        self.viewer = User.objects.create_user(username="queries_viewer", password="pw12345")
        UserRole.objects.create(user=self.viewer, role=self.role)

        self.log_owner = User.objects.create_user(username="queries_log_owner", password="pw12345")
        self.secret_question = "What is our unreleased product launch date?"
        QueryLog.objects.create(user=self.log_owner, question=self.secret_question, answer="Some answer.")

    def test_list_never_includes_question_text_even_with_view_content_permission(self):
        self.client.login(username="queries_viewer", password="pw12345")
        response = self.client.get("/api/admin/queries/")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["results"]), 1)
        self.assertNotIn("question", data["results"][0])
        self.assertNotIn("can_view_content", data)
        # Belt and suspenders: the secret text must not appear ANYWHERE
        # in the raw response body, not just be absent from the one
        # field we happen to check by key.
        self.assertNotIn(self.secret_question, response.content.decode())

    def test_content_text_search_param_is_inert(self):
        """`?q=<text>` used to search question/answer text for a "queries.view_content" holder - now a no-op for everyone, so it must never be used to infer/leak content either (e.g. via a 200 vs empty-results side channel matching a guessed word)."""
        self.client.login(username="queries_viewer", password="pw12345")
        response = self.client.get("/api/admin/queries/", {"q": "unreleased"})

        self.assertEqual(response.status_code, 200)
        # The filter never applies, so the log is still returned
        # unfiltered by content - not excluded, and not exposing its text.
        self.assertEqual(len(response.json()["results"]), 1)
        self.assertNotIn(self.secret_question, response.content.decode())

    def test_csv_export_never_includes_content_columns(self):
        self.client.login(username="queries_viewer", password="pw12345")
        response = self.client.get("/api/admin/queries/export.csv")

        self.assertEqual(response.status_code, 200)
        body = response.content.decode()
        self.assertNotIn(self.secret_question, body)
        self.assertNotIn("Question", body.splitlines()[0])

    def test_detail_endpoint_is_gone(self):
        log = QueryLog.objects.get(user=self.log_owner)
        self.client.login(username="queries_viewer", password="pw12345")
        response = self.client.get(f"/api/admin/queries/{log.id}/detail/")
        self.assertEqual(response.status_code, 404)


class DashboardCrossMemberQueryPrivacyTests(TestCase):
    """
    Regression test for a real bug: a Company Owner's own Dashboard
    (/api/dashboard/) builds its "recent questions" activity feed from
    stats_service._workspace_scope()'s WHOLE-organization QueryLog rows
    (correct for aggregate counts), but was then serializing every
    row's raw `question` text into the response regardless of whose
    row it was - leaking a plain Member's actual question to the Owner
    with no "queries.view_content" check at all (the same content
    boundary Admin > Queries enforces). Fixed by only ever showing the
    real text for the viewer's own row; every other member's row still
    shows up generically ("Asked a question") so the activity feed
    still reflects that something happened.
    """

    def setUp(self):
        from .services.organization_service import create_organization

        self.org_type = OrganizationType.objects.create(slug="dash_privacy_co", name="Company")
        self.owner = User.objects.create_user(username="dash_privacy_owner", password="pw12345")
        self.organization = create_organization(name="Dash Privacy Co", org_type=self.org_type, created_by=self.owner)

        self.member = User.objects.create_user(username="dash_privacy_member", password="pw12345")
        OrganizationMembership.objects.create(
            organization=self.organization, user=self.member, role=OrganizationMembership.Role.MEMBER,
        )

        self.secret_question = "Should we lay off the engineering team?"
        QueryLog.objects.create(
            user=self.member, organization=self.organization,
            question=self.secret_question, answer="Some answer.", confidence=80,
        )

    def test_owner_dashboard_never_exposes_a_members_question_text(self):
        self.client.login(username="dash_privacy_owner", password="pw12345")
        response = self.client.get("/api/dashboard/", HTTP_X_ORGANIZATION_SLUG=self.organization.slug)

        self.assertEqual(response.status_code, 200)
        self.assertNotIn(self.secret_question, response.content.decode())

        data = response.json()
        member_entries = [q for q in data["recent_questions"] if q["question"] is None]
        self.assertEqual(len(member_entries), 1)
        self.assertIn("Asked a question", [e["text"] for e in data["activity_feed"]])


class AdminCompanyWiseReportsTests(TestCase):
    """
    A platform Admin can pull ANY company's Reports (or their own
    Personal Workspace's) via an explicit ?organization=<slug>|personal
    query param on /api/reports/ and its CSV exports, without needing
    to first switch their active workspace to it - see reports_views.
    _resolve_report_organization()'s docstring for why (an Admin is
    rarely a real member of the company they need to audit). A
    non-admin - even that company's own real Owner - must never be
    able to use this param to see anything the ordinary header-based
    resolution wouldn't already show them; the override is only ever
    honored for permission_service.is_admin(). The company-wide usage
    export an Admin gets this way is metadata-only (no question/answer
    text), matching the same content boundary Admin > Queries enforces.
    """

    def setUp(self):
        from .models import ADMIN_ROLE_SLUG
        from .services.organization_service import create_organization

        self.org_type = OrganizationType.objects.create(slug="report_scope_co", name="Company")

        admin_role, _ = Role.objects.get_or_create(slug=ADMIN_ROLE_SLUG, defaults={"name": "Admin", "is_system": True})
        self.admin_user = User.objects.create_user(username="report_scope_admin", password="pw12345")
        UserRole.objects.create(user=self.admin_user, role=admin_role)

        reports_permission, _ = Permission.objects.get_or_create(codename="pages.reports", defaults={"name": "Reports"})
        outsider_role = Role.objects.create(slug="report_scope_outsider_role", name="Outsider")
        outsider_role.permissions.add(reports_permission)
        self.outsider = User.objects.create_user(username="report_scope_outsider", password="pw12345")
        UserRole.objects.create(user=self.outsider, role=outsider_role)

        self.owner = User.objects.create_user(username="report_scope_owner", password="pw12345")
        UserRole.objects.create(user=self.owner, role=outsider_role)
        self.organization = create_organization(name="Report Scope Co", org_type=self.org_type, created_by=self.owner)

        self.member = User.objects.create_user(username="report_scope_member", password="pw12345")
        UserRole.objects.create(user=self.member, role=outsider_role)
        OrganizationMembership.objects.create(organization=self.organization, user=self.member, role=OrganizationMembership.Role.MEMBER)

        self.secret_question = "Are we being acquired?"
        QueryLog.objects.create(
            user=self.member, organization=self.organization,
            question=self.secret_question, answer="Some answer.", confidence=70,
        )

        Document.objects.create(user=self.member, organization=self.organization, title="Board Deck.pdf", file="documents/board.pdf")

    def test_admin_can_view_any_companys_report_summary_via_override(self):
        self.client.login(username="report_scope_admin", password="pw12345")
        response = self.client.get(f"/api/reports/?organization={self.organization.slug}")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["document_count"], 1)
        self.assertEqual(data["question_count"], 1)

    def test_admin_company_usage_export_is_metadata_only_and_whole_team(self):
        self.client.login(username="report_scope_admin", password="pw12345")
        response = self.client.get(f"/api/reports/usage.csv?organization={self.organization.slug}")

        self.assertEqual(response.status_code, 200)
        body = response.content.decode()
        self.assertIn("report_scope_member", body)
        self.assertNotIn(self.secret_question, body)
        self.assertEqual(body.splitlines()[0].strip(), "Owner,Search Method,Confidence (%),Response Time (ms),Asked At")

    def test_admin_documents_export_covers_the_whole_company(self):
        self.client.login(username="report_scope_admin", password="pw12345")
        response = self.client.get(f"/api/reports/documents.csv?organization={self.organization.slug}")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Board Deck.pdf", response.content.decode())

    def test_admin_can_still_view_own_personal_workspace_via_override(self):
        Document.objects.create(user=self.admin_user, title="My Own Doc.pdf", file="documents/mine.pdf")

        self.client.login(username="report_scope_admin", password="pw12345")
        response = self.client.get("/api/reports/documents.csv?organization=personal")

        self.assertEqual(response.status_code, 200)
        self.assertIn("My Own Doc.pdf", response.content.decode())

    def test_non_admin_cannot_use_the_override_to_view_a_company_they_do_not_belong_to(self):
        self.client.login(username="report_scope_outsider", password="pw12345")
        response = self.client.get(f"/api/reports/?organization={self.organization.slug}")

        # Override ignored (outsider isn't Admin) - falls back to the
        # header-based resolution, which (no header, outsider has no
        # membership) resolves to their own Personal Workspace, never
        # the named company's data.
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["document_count"], 0)

    def test_the_companys_own_owner_gets_no_special_treatment_from_the_override_either(self):
        """
        The override is admin-only, not membership-based - Owner isn't
        is_admin(), so ?organization=<their own company> is ignored and
        falls back to the ordinary header-based resolution. Headerless,
        that resolves to the Owner's own company anyway (a COMPANY
        account_type - see resolve_request_organization()'s docstring -
        defaults to its first/only organization, not Personal
        Workspace) - and the Owner can view it, same as if they'd sent
        the header explicitly (see user_can_view_org_analytics(): an
        Owner sees their own company's Reports, the same org-management
        surface rank they already hold everywhere else).
        """
        self.client.login(username="report_scope_owner", password="pw12345")
        response = self.client.get(f"/api/reports/?organization={self.organization.slug}")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["document_count"], 1)
        self.assertEqual(data["question_count"], 1)

    def test_owner_usage_export_is_whole_team_and_metadata_only(self):
        """Once inside their own company, an Owner's Usage Report is the whole team's usage (not just their own rows) and never includes question/answer text - matching Documents/AI Task Runs' existing whole-team behavior and Admin > Queries' content boundary."""
        self.client.login(username="report_scope_owner", password="pw12345")
        response = self.client.get(f"/api/reports/usage.csv?organization={self.organization.slug}")

        self.assertEqual(response.status_code, 200)
        body = response.content.decode()
        self.assertIn("report_scope_member", body)
        self.assertNotIn(self.secret_question, body)
        self.assertEqual(body.splitlines()[0].strip(), "Owner,Search Method,Confidence (%),Response Time (ms),Asked At")

    def test_a_member_explicitly_restricted_by_their_owner_cannot_view_the_companys_reports(self):
        """
        Reports for a Member's own company is no longer Owner-only by
        default (see MemberGrantedAnalyticsReportsAccessTests - a
        Member has Analytics/Reports access the same as every other
        FEATURE_CODES entry unless their Owner restricts it), but the
        Owner's own explicit restriction (set_member_disabled_features)
        still applies. Uses the X-Organization-Slug header (not
        ?organization=, which only ever does anything for is_admin())
        to actually put the company in scope for this request - the
        member's account_type is still "personal" here (this test
        creates the OrganizationMembership row directly, bypassing the
        real registration flow that would force account_type=COMPANY),
        so a headerless request would resolve to their empty Personal
        Workspace instead and 200 there for an unrelated reason,
        without ever exercising the check this test exists to cover.
        """
        from .services.org_member_feature_service import set_member_disabled_features

        owner_membership = OrganizationMembership.objects.get(organization=self.organization, user=self.owner)
        member_membership = OrganizationMembership.objects.get(organization=self.organization, user=self.member)
        set_member_disabled_features(self.organization, owner_membership, member_membership, ["reports"])

        self.client.login(username="report_scope_member", password="pw12345")
        response = self.client.get("/api/reports/", HTTP_X_ORGANIZATION_SLUG=self.organization.slug)

        self.assertEqual(response.status_code, 403)


class BackfillFreePlanMigrationTests(TestCase):
    """
    RAG.migrations.0055_backfill_free_plan_for_organizations - every
    active organization with no Subscription row at all gets the
    built-in Free plan; an organization that already has a (possibly
    different, deliberately assigned) plan is left completely alone.
    """

    def test_backfill_assigns_free_plan_only_to_organizations_missing_one(self):
        from importlib import import_module

        from .models import Subscription
        from .services.billing_service import FREE_PLAN_SLUG, create_plan, get_or_create_free_plan
        from .services.organization_service import create_organization

        backfill_module = import_module("RAG.migrations.0055_backfill_free_plan_for_organizations")

        org_type = OrganizationType.objects.create(slug="backfill_test_co", name="Company")
        creator = User.objects.create_user(username="backfill_test_creator", password="pw")

        # Simulates an organization created BEFORE the auto-assignment
        # fix existed: create it, then delete the Subscription
        # create_organization() itself just assigned, so it's back to
        # the old "no row = unlimited" state this migration targets.
        without_plan = create_organization(name="Backfill No Plan Co", org_type=org_type, created_by=creator)
        Subscription.objects.filter(organization=without_plan).delete()

        other_creator = User.objects.create_user(username="backfill_test_creator_2", password="pw")
        with_plan = create_organization(name="Backfill Has Plan Co", org_type=org_type, created_by=other_creator)
        custom_plan = create_plan(name="Backfill Custom Plan", included_credits=500)
        from .services.billing_service import assign_plan
        assign_plan(with_plan, custom_plan, other_creator)

        backfill_module.backfill_free_plan(apps=None, schema_editor=None)

        without_plan_subscription = Subscription.objects.get(organization=without_plan)
        self.assertEqual(without_plan_subscription.plan.slug, FREE_PLAN_SLUG)

        with_plan_subscription = Subscription.objects.get(organization=with_plan)
        self.assertEqual(with_plan_subscription.plan_id, custom_plan.id)

        # Idempotent - running it again changes nothing further.
        backfill_module.backfill_free_plan(apps=None, schema_editor=None)
        self.assertEqual(Subscription.objects.get(organization=without_plan).plan.slug, FREE_PLAN_SLUG)
        self.assertEqual(get_or_create_free_plan().subscriptions.filter(organization=without_plan).count(), 1)
