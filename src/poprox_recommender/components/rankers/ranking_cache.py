import copy
import hashlib
import os
import pickle
import threading
from collections import OrderedDict
from datetime import datetime
from typing import Optional

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:  # pragma: no cover - boto3 is optional in local dev
    boto3 = None
    ClientError = Exception

from poprox_concepts import CandidateSet
from poprox_concepts.domain import RecommendationList

_memory_cache: Optional["MemoryRankingCache"] = None


def _generate_cache_key(
    profile_id: str,
    candidate_articles: CandidateSet,
    model_version: Optional[str],
) -> str:
    """
    Generate a deterministic cache key from profile ID and candidate set.
    """
    article_ids = sorted(str(art.article_id) for art in candidate_articles.articles)
    articles_hash = hashlib.sha256(",".join(article_ids).encode()).hexdigest()[:16]
    version = model_version or "default"
    return f"{profile_id}_{articles_hash}_{version}"


def _clone_ranking_output(ranking_output: tuple[RecommendationList, str, str, dict, dict]):
    """
    Create a deep copy of the ranking output so cached values remain immutable.
    """
    recommendations, user_model, request_id, llm_metrics, component_metrics = ranking_output
    cloned_recommendations = RecommendationList(articles=copy.deepcopy(recommendations.articles))
    return (
        cloned_recommendations,
        user_model,
        request_id,
        copy.deepcopy(llm_metrics),
        copy.deepcopy(component_metrics),
    )


class MemoryRankingCache:
    """
    In-memory LRU cache for ranking outputs shared within a single Lambda container.
    """

    def __init__(self, max_entries: int = 128):
        self.max_entries = max(max_entries, 1)
        self._store: "OrderedDict[str, tuple[RecommendationList, str, str, dict, dict]]" = OrderedDict()
        self._lock = threading.Lock()

    def get_cached_ranking(
        self,
        profile_id: str,
        candidate_articles: CandidateSet,
        model_version: Optional[str] = None,
    ) -> Optional[tuple[RecommendationList, str, str, dict, dict]]:
        cache_key = _generate_cache_key(profile_id, candidate_articles, model_version)
        with self._lock:
            cached = self._store.get(cache_key)
            if cached is None:
                return None
            self._store.move_to_end(cache_key)
            return _clone_ranking_output(cached)

    def save_ranking(
        self,
        profile_id: str,
        candidate_articles: CandidateSet,
        ranking_output: tuple[RecommendationList, str, str, dict, dict],
        model_version: Optional[str] = None,
    ) -> str:
        cache_key = _generate_cache_key(profile_id, candidate_articles, model_version)
        with self._lock:
            self._store[cache_key] = _clone_ranking_output(ranking_output)
            self._store.move_to_end(cache_key)
            while len(self._store) > self.max_entries:
                self._store.popitem(last=False)
        return cache_key


class S3RankingCache:
    """
    S3-based cache for LLM ranking outputs to ensure consistency across pipelines.
    """

    def __init__(
        self,
        bucket_name: Optional[str] = None,
        prefix: str = "ranking-cache/",
        model_version: str = "gpt-4.1-mini-2025-04-14",
    ):
        if boto3 is None:
            raise ImportError("boto3 is required for S3 ranking cache. Install with: pip install boto3")

        self.s3 = boto3.client("s3")
        self.bucket = bucket_name or os.getenv(
            "RANKING_CACHE_BUCKET",
            os.getenv("PERSISTENCE_BUCKET", "poprox-default-recommender-pipeline-data-prod"),
        )
        self.prefix = prefix
        self.model_version = model_version

    def _get_s3_key(self, cache_key: str) -> str:
        date_str = datetime.now().strftime("%Y-%m-%d")
        return f"{self.prefix}{date_str}/ranking_{cache_key}.pkl"

    def get_cached_ranking(
        self,
        profile_id: str,
        candidate_articles: CandidateSet,
        model_version: Optional[str] = None,
    ) -> Optional[tuple[RecommendationList, str, str, dict, dict]]:
        cache_key = _generate_cache_key(profile_id, candidate_articles, model_version or self.model_version)
        s3_key = self._get_s3_key(cache_key)
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=s3_key)
            return pickle.loads(response["Body"].read())
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                return None
            print(f"Warning: Failed to retrieve cache from S3: {e}")
            return None

    def save_ranking(
        self,
        profile_id: str,
        candidate_articles: CandidateSet,
        ranking_output: tuple[RecommendationList, str, str, dict, dict],
        model_version: Optional[str] = None,
    ) -> str:
        cache_key = _generate_cache_key(profile_id, candidate_articles, model_version or self.model_version)
        s3_key = self._get_s3_key(cache_key)
        try:
            self.s3.put_object(
                Bucket=self.bucket,
                Key=s3_key,
                Body=pickle.dumps(ranking_output),
                ContentType="application/octet-stream",
                Metadata={
                    "profile_id": profile_id,
                    "cache_key": cache_key,
                    "model_version": model_version or self.model_version,
                    "cached_at": datetime.now().isoformat(),
                },
            )
        except ClientError as e:
            print(f"Warning: Failed to save cache to S3: {e}")
        return cache_key


# Backwards compatibility with previous import path/tests
RankingCacheManager = S3RankingCache


class RankingCacheChain:
    """
    Chain multiple cache backends. Uses the in-memory cache first, then falls back to S3.
    """

    def __init__(self, primary: MemoryRankingCache, secondary: Optional[S3RankingCache] = None):
        self.primary = primary
        self.secondary = secondary

    def get_cached_ranking(
        self,
        profile_id: str,
        candidate_articles: CandidateSet,
        model_version: Optional[str] = None,
    ) -> Optional[tuple[RecommendationList, str, str, dict, dict]]:
        result = self.primary.get_cached_ranking(profile_id, candidate_articles, model_version)
        if result is not None:
            return result

        if self.secondary is None:
            return None

        secondary_result = self.secondary.get_cached_ranking(profile_id, candidate_articles, model_version)
        if secondary_result is not None:
            self.primary.save_ranking(profile_id, candidate_articles, secondary_result, model_version)
        return secondary_result

    def save_ranking(
        self,
        profile_id: str,
        candidate_articles: CandidateSet,
        ranking_output: tuple[RecommendationList, str, str, dict, dict],
        model_version: Optional[str] = None,
    ) -> str:
        cache_key = self.primary.save_ranking(profile_id, candidate_articles, ranking_output, model_version)
        if self.secondary is not None:
            self.secondary.save_ranking(profile_id, candidate_articles, ranking_output, model_version)
        return cache_key


def _get_memory_cache(max_entries: Optional[int] = None) -> MemoryRankingCache:
    global _memory_cache
    if _memory_cache is None:
        size = max_entries or int(os.getenv("RANKING_CACHE_MEMORY_SIZE", "128"))
        _memory_cache = MemoryRankingCache(max_entries=max(size, 1))
    return _memory_cache


def _determine_cache_mode() -> str:
    mode = os.getenv("RANKING_CACHE_MODE")
    if mode:
        return mode.strip().lower()

    enabled = os.getenv("RANKING_CACHE_ENABLED")
    if enabled is None:
        return "memory"

    value = enabled.strip().lower()
    if value in {"0", "false", "no", "off"}:
        return "off"
    return "memory+s3"


def get_ranking_cache_manager(model_version: str = "gpt-4.1-mini-2025-04-14"):
    """
    Factory function to get ranking cache manager based on environment.
    """
    mode = _determine_cache_mode()
    if mode in {"off", "none"}:
        return None

    memory_cache = _get_memory_cache()
    if mode == "memory":
        return memory_cache

    s3_cache: Optional[S3RankingCache] = None
    if mode in {"s3", "memory+s3"}:
        try:
            bucket = os.getenv(
                "RANKING_CACHE_BUCKET",
                os.getenv("PERSISTENCE_BUCKET", "poprox-default-recommender-pipeline-data-prod"),
            )
            prefix = os.getenv("RANKING_CACHE_PREFIX", "ranking-cache/")
            s3_cache = S3RankingCache(bucket, prefix, model_version)
        except ImportError:
            s3_cache = None

    if mode == "s3":
        return s3_cache

    if s3_cache is None:
        return memory_cache

    return RankingCacheChain(memory_cache, s3_cache)
