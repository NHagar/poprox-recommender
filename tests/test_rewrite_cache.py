import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from poprox_recommender.components.rewriters.openai_rewriter import LLMRewriter, LLMRewriterConfig
from poprox_recommender.components.rewriters.rewrite_cache import (
    MemoryRewriteCache,
    get_rewrite_cache_manager,
    hash_user_model,
    reset_rewrite_cache,
)

poprox_concepts = pytest.importorskip("poprox_concepts")
from poprox_concepts import Article  # noqa: E402
from poprox_concepts.domain import RecommendationList  # noqa: E402


def test_memory_rewrite_cache_roundtrip():
    cache = MemoryRewriteCache(max_entries=4, ttl_seconds=60)
    article_id = "article-123"
    user_hash = hash_user_model("sample user model")

    cache.save_rewrite(
        article_id=article_id,
        user_model_hash=user_hash,
        original_headline="Original Headline",
        rewritten_headline="Rewritten Headline",
        pipeline_name="test-pipeline",
    )

    entry = cache.get_cached_rewrite(
        article_id=article_id,
        user_model_hash=user_hash,
        original_headline="Original Headline",
        pipeline_name="test-pipeline",
    )
    assert entry is not None
    assert entry.rewritten_headline == "Rewritten Headline"


def test_memory_rewrite_cache_headline_mismatch_invalidates():
    cache = MemoryRewriteCache(max_entries=4, ttl_seconds=60)
    article_id = "article-456"
    user_hash = hash_user_model("user model")

    cache.save_rewrite(
        article_id=article_id,
        user_model_hash=user_hash,
        original_headline="Correct Headline",
        rewritten_headline="Rewritten",
        pipeline_name="test-pipeline",
    )

    entry = cache.get_cached_rewrite(
        article_id=article_id,
        user_model_hash=user_hash,
        original_headline="Different Headline",
        pipeline_name="test-pipeline",
    )
    assert entry is None


def test_get_rewrite_cache_manager_disabled():
    with patch.dict(os.environ, {"REWRITE_CACHE_ENABLED": "false"}):
        cache = get_rewrite_cache_manager()
        assert cache is None


@patch("poprox_recommender.components.rewriters.openai_rewriter.is_persistence_enabled", return_value=False)
def test_llm_rewriter_uses_cached_rewrites(_mock_persistence):
    reset_rewrite_cache()

    user_model = """Topics the user has shown interest in (from most to least):
Topic One, Topic Two

Topics the user has clicked on (from most to least):
Topic One, Topic Three

Headlines of articles the user has clicked on recently:
Sample headline
"""

    article_id = uuid4()
    request_id = "request-xyz"

    def make_ranker_output():
        article = Article(
            article_id=article_id,
            headline="Original headline",
            body="Body text for the article.",
        )
        recommendations = RecommendationList(articles=[article])
        return (recommendations, user_model, request_id, {}, {})

    class DummyAsyncClient:
        def __init__(self, parse_mock):
            self.responses = SimpleNamespace(parse=parse_mock)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    parse_mock = AsyncMock()
    parse_mock.return_value = SimpleNamespace(
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        output_parsed=SimpleNamespace(headline="Rewritten headline"),
    )

    def async_openai_factory(*args, **kwargs):
        return DummyAsyncClient(parse_mock)

    with patch("poprox_recommender.components.rewriters.openai_rewriter.openai.AsyncOpenAI", side_effect=async_openai_factory):
        rewriter = LLMRewriter(LLMRewriterConfig(enable_cache=True))

        first_output = make_ranker_output()
        first_result = rewriter(first_output)
        assert parse_mock.await_count == 1
        assert first_result.articles[0].headline == "Rewritten headline"

        second_output = make_ranker_output()
        second_result = rewriter(second_output)

        # Ensure we reused cached rewrite and did not make another LLM call
        assert parse_mock.await_count == 1
        assert second_result.articles[0].headline == "Rewritten headline"
