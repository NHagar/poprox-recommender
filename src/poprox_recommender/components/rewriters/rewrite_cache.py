import hashlib
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RewriteCacheEntry:
    """
    Cached rewrite result for a single article within a request context.
    """

    rewritten_headline: str
    original_headline: str
    user_model_hash: str
    article_id: str
    cached_at: float


def _generate_cache_key(request_id: str, article_id: str, user_model_hash: str) -> str:
    return f"{request_id}:{article_id}:{user_model_hash}"


def hash_user_model(user_model: str | None) -> str:
    """
    Generate a stable hash for the user model string. Falls back to hashing an empty string.
    """
    if not user_model:
        return "0" * 16
    return hashlib.sha256(user_model.encode("utf-8")).hexdigest()[:16]


class MemoryRewriteCache:
    """
    Simple in-memory LRU cache for rewrite results shared within a Lambda container.
    """

    def __init__(self, max_entries: int = 512, ttl_seconds: int = 900):
        self.max_entries = max(1, max_entries)
        self.ttl_seconds = max(0, ttl_seconds)
        self._store: "OrderedDict[str, RewriteCacheEntry]" = OrderedDict()
        self._lock = threading.Lock()

    def _is_expired(self, entry: RewriteCacheEntry) -> bool:
        if self.ttl_seconds == 0:
            return False
        return time.time() - entry.cached_at > self.ttl_seconds

    def get_cached_rewrite(
        self,
        request_id: str,
        article_id: str,
        user_model_hash: str,
        original_headline: str,
    ) -> Optional[RewriteCacheEntry]:
        cache_key = _generate_cache_key(request_id, article_id, user_model_hash)
        with self._lock:
            entry = self._store.get(cache_key)
            if entry is None:
                return None

            if self._is_expired(entry) or entry.original_headline != original_headline:
                self._store.pop(cache_key, None)
                return None

            self._store.move_to_end(cache_key)
            return entry

    def save_rewrite(
        self,
        request_id: str,
        article_id: str,
        user_model_hash: str,
        original_headline: str,
        rewritten_headline: str,
    ) -> str:
        cache_key = _generate_cache_key(request_id, article_id, user_model_hash)
        entry = RewriteCacheEntry(
            rewritten_headline=rewritten_headline,
            original_headline=original_headline,
            user_model_hash=user_model_hash,
            article_id=article_id,
            cached_at=time.time(),
        )
        with self._lock:
            self._store[cache_key] = entry
            self._store.move_to_end(cache_key)
            while len(self._store) > self.max_entries:
                self._store.popitem(last=False)
        return cache_key

    def clear(self):
        with self._lock:
            self._store.clear()


_memory_cache: Optional[MemoryRewriteCache] = None


def _get_memory_cache() -> MemoryRewriteCache:
    global _memory_cache
    if _memory_cache is None:
        size = int(os.getenv("REWRITE_CACHE_MEMORY_SIZE", "512"))
        ttl = int(os.getenv("REWRITE_CACHE_TTL_SECONDS", "900"))
        _memory_cache = MemoryRewriteCache(max_entries=max(1, size), ttl_seconds=max(0, ttl))
    return _memory_cache


def _determine_cache_mode() -> str:
    mode = os.getenv("REWRITE_CACHE_MODE")
    if mode:
        return mode.strip().lower()

    enabled = os.getenv("REWRITE_CACHE_ENABLED")
    if enabled is None:
        return "memory"

    value = enabled.strip().lower()
    if value in {"0", "false", "no", "off"}:
        return "off"
    return "memory"


def get_rewrite_cache_manager() -> Optional[MemoryRewriteCache]:
    """
    Get a rewrite cache instance based on environment configuration.
    """
    mode = _determine_cache_mode()
    if mode in {"off", "none"}:
        return None

    # Only memory mode is supported today; fall back to memory for unknown modes.
    return _get_memory_cache()


def reset_rewrite_cache():
    """
    Reset in-memory cache for testing.
    """
    cache = get_rewrite_cache_manager()
    if cache is not None:
        cache.clear()
