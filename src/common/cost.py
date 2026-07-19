"""Thread-safe running tally of OpenRouter LLM call costs.

Accumulates the per-call cost reported by OpenRouter (parsed from the
response headers / body by ``src.transform.client``) and emits both a
per-call INFO line and a final ``COST SUMMARY`` line. The summary is
emitted from a ``finally`` block in ``src.main.main`` so it lands in
``logs/pipeline.log`` regardless of whether the pipeline succeeded or
failed.

The accumulator is process-global and protected by a single lock, so it
is correct under the parallel runner's ``ThreadPoolExecutor``.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class _Totals:
    total_cost: float = 0.0
    calls: int = 0
    by_page: dict[int, float] = field(default_factory=dict)


_lock = threading.Lock()
_totals = _Totals()


def record_call(page: int, cost: float | None) -> None:
    """Add one LLM call's cost to the running tally and log it.

    ``cost=None`` (provider didn't return a parseable cost) is treated as
    ``0.0`` for the tally and logged as a warning so the gap is visible
    in the logfile rather than silently swallowing the call.
    """
    parsed = float(cost) if cost is not None else 0.0
    with _lock:
        _totals.calls += 1
        _totals.total_cost += parsed
        _totals.by_page[page] = _totals.by_page.get(page, 0.0) + parsed
        page_running = _totals.by_page[page]
        total = _totals.total_cost
        calls = _totals.calls

    if cost is None:
        logger.warning(
            "page %d call %d: OpenRouter cost unavailable (counted as $0; "
            "total=$%.6f calls=%d)",
            page,
            calls,
            total,
            calls,
        )
    else:
        logger.info(
            "page %d cost=$%.6f (page_running=$%.6f total=$%.6f calls=%d)",
            page,
            parsed,
            page_running,
            total,
            calls,
        )


def log_summary() -> None:
    """Emit the final ``COST SUMMARY`` line. Safe to call from a finally.

    Reads the totals under the lock so the printed numbers are
    internally consistent. No-op if no calls were recorded.
    """
    with _lock:
        total = _totals.total_cost
        calls = _totals.calls
        pages = len(_totals.by_page)
    if calls == 0:
        logger.info("COST SUMMARY no LLM calls recorded")
        return
    logger.info(
        "COST SUMMARY total=$%.6f across %d LLM calls (%d pages)",
        total,
        calls,
        pages,
    )


def reset() -> None:
    """Clear the accumulator. Intended for tests."""
    global _totals
    with _lock:
        _totals = _Totals()