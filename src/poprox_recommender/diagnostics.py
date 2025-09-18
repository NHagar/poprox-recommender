"""Helpers for emitting timing and optional profiling information to the logs."""

from __future__ import annotations

import contextlib
import cProfile
import io
import os
import sys
import time
from typing import Iterator

import structlog
from structlog.stdlib import BoundLogger

try:  # pragma: no cover - platform-specific
    import resource
except ImportError:  # pragma: no cover - Windows fallback
    resource = None  # type: ignore[assignment]

_logger = structlog.get_logger(__name__)


def _bool_env(var_name: str) -> bool:
    value = os.getenv(var_name, "")
    return value.lower() in {"1", "true", "yes", "on"}


def _maxrss_mb() -> float | None:
    if resource is None:  # pragma: no cover - platform-specific
        return None

    usage = resource.getrusage(resource.RUSAGE_SELF)
    # Linux reports ru_maxrss in KiB, macOS reports bytes.
    scale = 1024 if sys.platform != "darwin" else 1
    return usage.ru_maxrss / (scale * 1024)


@contextlib.contextmanager
def timed_section(
    section: str,
    *,
    log_start: bool = True,
    profiling_env: str = "POPROX_PROFILE_WARMUP",
    profile_lines: int = 20,
    profile_sort: str = "cumtime",
    logger: BoundLogger | None = None,
    **log_fields: object,
) -> Iterator[None]:
    """Log the duration of a code section and optionally emit cProfile stats.

    Enable profiling for a section by setting the ``profiling_env`` environment
    variable (defaults to ``POPROX_PROFILE_WARMUP``) to a truthy value. When
    enabled, the top ``profile_lines`` entries sorted by ``profile_sort`` are
    emitted to the logs. Both parameters can be overridden with environment
    variables (``{profiling_env}_LINES`` and ``{profiling_env}_SORT``).
    """

    if logger is None:
        log = _logger
    else:
        log = logger

    start = time.perf_counter()
    peak_before = _maxrss_mb()

    if log_start:
        log.info("warmup_section.start", **log_fields)

    env_enabled = _bool_env(profiling_env)
    lines_to_show = profile_lines
    sort_order = profile_sort
    if env_enabled:
        lines_override = os.getenv(f"{profiling_env}_LINES")
        sort_override = os.getenv(f"{profiling_env}_SORT")

        if lines_override is not None:
            try:
                lines_to_show = int(lines_override)
            except ValueError:
                log.warning(
                    "warmup_section.profile_config_invalid_lines",
                    value=lines_override,
                )

        if sort_override:
            sort_order = sort_override

        profiler = cProfile.Profile()
        profiler.enable()
    else:
        profiler = None

    try:
        yield
    finally:
        if profiler is not None:
            profiler.disable()

        elapsed = time.perf_counter() - start
        peak_after = _maxrss_mb()
        mem_delta = None
        if peak_before is not None and peak_after is not None:
            mem_delta = max(0.0, peak_after - peak_before)

        log_kwargs: dict[str, object] = {
            "event": section,
            "elapsed_ms": round(elapsed * 1000, 2),
            **log_fields,
        }

        if peak_after is not None:
            log_kwargs["peak_rss_mb"] = round(peak_after, 2)
        if mem_delta is not None:
            log_kwargs["delta_peak_rss_mb"] = round(mem_delta, 2)

        log.info("warmup_section.complete", **log_kwargs)

        if profiler is not None:
            stats_stream = io.StringIO()
            try:
                pstats = __import__("pstats")
                pstats.Stats(profiler, stream=stats_stream).sort_stats(sort_order).print_stats(lines_to_show)
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("warmup_section.profile_failed", exc_info=exc)
            else:
                stats_stream.seek(0)
                log.info(
                    "warmup_section.profile",
                    profile_sort=sort_order,
                    lines=lines_to_show,
                    stats=stats_stream.read(),
                    **log_fields,
                )
