import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from .base import PersistenceManager
from .local import LocalPersistenceManager

# Default S3 bucket for persistence
DEFAULT_PERSISTENCE_BUCKET = "poprox-default-recommender-pipeline-data-prod"

logger = logging.getLogger(__name__)

_executor: ThreadPoolExecutor | None = None


def _persistence_mode() -> str:
    return os.getenv("PERSISTENCE_MODE", "async").strip().lower()


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        workers = int(os.getenv("PERSISTENCE_WORKERS", "2"))
        _executor = ThreadPoolExecutor(max_workers=max(workers, 1))
    return _executor


def get_persistence_manager() -> PersistenceManager:
    """
    Factory function to get the appropriate persistence manager based on environment.

    Returns:
        PersistenceManager: LocalPersistenceManager for local dev, S3PersistenceManager for Lambda
    """
    backend_override = os.getenv("PERSISTENCE_BACKEND")
    backend = backend_override.lower() if backend_override else None

    if backend not in (None, "auto", "s3", "local"):
        raise ValueError(f"Unsupported PERSISTENCE_BACKEND value: {backend_override}")

    use_s3 = False
    if backend == "s3":
        use_s3 = True
    elif backend == "local":
        use_s3 = False
    else:
        # Auto mode defaults to S3 when running in Lambda
        use_s3 = bool(os.getenv("AWS_LAMBDA_FUNCTION_NAME"))

    if use_s3:
        from .s3 import S3PersistenceManager

        bucket = os.getenv("PERSISTENCE_BUCKET", DEFAULT_PERSISTENCE_BUCKET)
        prefix = os.getenv("PERSISTENCE_PREFIX", "pipeline-outputs/")
        return S3PersistenceManager(bucket, prefix)

    persistence_path = os.getenv("PERSISTENCE_PATH", "./data/pipeline_outputs")
    return LocalPersistenceManager(persistence_path)


def schedule_pipeline_save(description: str, save_callable: Callable[[], Any]) -> Any | None:
    """
    Schedule a persistence save operation according to the configured mode.

    Modes:
      - off: skip persistence entirely
      - sync: execute immediately and return the save result
      - async (default): run on a background executor and return None
    """
    mode = _persistence_mode()

    if mode == "off":
        logger.info("Skipping persistence (%s) because PERSISTENCE_MODE=off", description)
        return None

    if mode == "sync":
        result = save_callable()
        logger.info("Completed synchronous persistence (%s)", description)
        return result

    executor = _get_executor()
    future = executor.submit(save_callable)

    def _log_completion(fut):
        try:
            session_id = fut.result()
            logger.info("Completed asynchronous persistence (%s): %s", description, session_id)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Asynchronous persistence failed (%s): %s", description, exc, exc_info=exc)

    future.add_done_callback(_log_completion)
    return None


def is_persistence_enabled() -> bool:
    """
    Determine whether persistence writes are enabled under the current mode.
    """
    return _persistence_mode() != "off"


__all__ = [
    "PersistenceManager",
    "LocalPersistenceManager",
    "get_persistence_manager",
    "DEFAULT_PERSISTENCE_BUCKET",
    "schedule_pipeline_save",
    "is_persistence_enabled",
]
