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


_component_timing_patch_applied = False


def component_timing_enabled() -> bool:
    """Check if component-level timing instrumentation should be enabled."""

    return _bool_env("POPROX_COMPONENT_TIMING")


def _describe_component(comp: object) -> str:
    """Return a descriptive name for a pipeline component callable."""

    module = getattr(comp, "__module__", None)
    qualname = getattr(comp, "__qualname__", None)
    if qualname is None and hasattr(comp, "__class__"):
        qualname = getattr(comp.__class__, "__qualname__", comp.__class__.__name__)

    if module:
        return f"{module}.{qualname}" if qualname else module
    return qualname or repr(comp)


def enable_pipeline_component_timing() -> None:
    """Monkey-patch LensKit's pipeline runner to log per-component durations."""

    global _component_timing_patch_applied

    if _component_timing_patch_applied or not component_timing_enabled():
        return

    try:
        from lenskit.diagnostics import PipelineError
        from lenskit.logging import trace
        from lenskit.pipeline.components import component_inputs
        from lenskit.pipeline.runner import DeferredRun, PipelineRunner
        from lenskit.pipeline.types import Lazy, is_compatible_data
    except Exception:  # pragma: no cover - defensive, lenskit optional in tests
        return

    from typing import get_args, get_origin

    def instrumented_run_component(self: PipelineRunner, name: str, comp, required: bool) -> None:
        in_data = {}
        log = self.log.bind(node=name)
        trace(log, "processing inputs")
        inputs = component_inputs(comp, warn_on_missing=False)
        wiring = self.pipe.node_input_connections(name)

        for iname, itype in inputs.items():
            ilog = log.bind(input_name=iname, input_type=itype)
            trace(ilog, "resolving input")
            snode = None
            if src := wiring.get(iname, None):
                trace(ilog, "resolving from wiring")
                snode = self.pipe.node(src)

            lazy = False
            if itype is not None:
                origin = get_origin(itype)
                if origin is Lazy:
                    lazy = True
                    (itype,) = get_args(itype)

            if snode is None:
                ival = None
            else:
                if required and itype:
                    ireq = not is_compatible_data(None, itype)
                else:
                    ireq = False

                if lazy:
                    ival = DeferredRun(self, iname, name, snode, required=ireq, data_type=itype)
                else:
                    ival = self.run(snode, required=ireq)

            if (
                ival is None
                and itype
                and not lazy
                and not is_compatible_data(None, itype)
                and not required
            ):
                return None

            if itype and not lazy and not is_compatible_data(ival, itype):
                if ival is None:
                    raise PipelineError(
                        f"no data available for required input ❬{iname}❭ on component ❬{name}❭"
                    )
                raise TypeError(
                    f"input ❬{iname}❭ on component ❬{name}❭ has invalid type {type(ival)} (expected {itype})"
                )

            in_data[iname] = ival

        trace(log, "running component", component=comp)
        component_type = _describe_component(comp)
        with timed_section(
            f"{self.pipe.name}.{name}",
            event_prefix="pipeline_component",
            profiling_env="POPROX_PROFILE_COMPONENT",
            logger=log,
            log_start=False,
            pipeline=self.pipe.name,
            component=name,
            component_type=component_type,
        ):
            self.state[name] = comp(**in_data)

    PipelineRunner._run_component = instrumented_run_component  # type: ignore[assignment]
    _component_timing_patch_applied = True

@contextlib.contextmanager
def timed_section(
    section: str,
    *,
    log_start: bool = True,
    event_prefix: str = "warmup_section",
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

    The emitted log event names default to the ``warmup_section.*`` prefix and
    can be customized with ``event_prefix``.
    """

    if logger is None:
        log = _logger
    else:
        log = logger

    prefix = event_prefix or "warmup_section"
    start_event = f"{prefix}.start"
    complete_event = f"{prefix}.complete"
    profile_event = f"{prefix}.profile"
    profile_config_event = f"{prefix}.profile_config_invalid_lines"
    profile_failed_event = f"{prefix}.profile_failed"

    start = time.perf_counter()
    peak_before = _maxrss_mb()

    if log_start:
        start_fields = {**log_fields, "section": section}
        log.info(start_event, **start_fields)

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
                    profile_config_event,
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
            **log_fields,
            "section": section,
            "elapsed_ms": round(elapsed * 1000, 2),
        }

        if peak_after is not None:
            log_kwargs["peak_rss_mb"] = round(peak_after, 2)
        if mem_delta is not None:
            log_kwargs["delta_peak_rss_mb"] = round(mem_delta, 2)

        log.info(complete_event, **log_kwargs)

        if profiler is not None:
            stats_stream = io.StringIO()
            try:
                pstats = __import__("pstats")
                pstats.Stats(profiler, stream=stats_stream).sort_stats(sort_order).print_stats(lines_to_show)
            except Exception as exc:  # pragma: no cover - defensive
                log.warning(profile_failed_event, exc_info=exc)
            else:
                stats_stream.seek(0)
                profile_fields = {
                    **log_fields,
                    "section": section,
                    "profile_sort": sort_order,
                    "lines": lines_to_show,
                    "stats": stats_stream.read(),
                }
                log.info(profile_event, **profile_fields)
