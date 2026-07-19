"""Unit tests for src.common.cost (thread-safe cost accumulator)."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from src.common import cost as cost_module
from src.common.cost import log_summary, record_call


@pytest.fixture(autouse=True)
def _reset_totals():
    cost_module.reset()
    yield
    cost_module.reset()


class TestRecordCall:
    def test_records_single_call(self):
        record_call(page=5, cost=0.0123)

        assert cost_module._totals.calls == 1
        assert cost_module._totals.total_cost == pytest.approx(0.0123)
        assert cost_module._totals.by_page[5] == pytest.approx(0.0123)

    def test_accumulates_across_pages_and_attempts(self):
        record_call(page=1, cost=0.01)
        record_call(page=2, cost=0.02)
        record_call(page=2, cost=0.03)  # retry on same page

        assert cost_module._totals.calls == 3
        assert cost_module._totals.total_cost == pytest.approx(0.06)
        assert cost_module._totals.by_page[1] == pytest.approx(0.01)
        assert cost_module._totals.by_page[2] == pytest.approx(0.05)

    def test_none_cost_treated_as_zero(self, caplog):
        with caplog.at_level(logging.WARNING, logger="src.common.cost"):
            record_call(page=1, cost=None)

        assert cost_module._totals.calls == 1
        assert cost_module._totals.total_cost == 0.0
        assert any(
            "cost unavailable" in r.getMessage() and r.levelno == logging.WARNING
            for r in caplog.records
        )

    def test_emits_per_call_info_line(self, caplog):
        with caplog.at_level(logging.INFO, logger="src.common.cost"):
            record_call(page=7, cost=0.0123)

        infos = [
            r for r in caplog.records
            if r.levelno == logging.INFO and "page 7 cost=" in r.getMessage()
        ]
        assert len(infos) == 1
        msg = infos[0].getMessage()
        # Per-page and running totals both reported.
        assert "page_running=$0.012300" in msg
        assert "total=$0.012300" in msg
        assert "calls=1" in msg


class TestLogSummary:
    def test_summary_reports_total_calls_pages(self, caplog):
        with caplog.at_level(logging.INFO, logger="src.common.cost"):
            record_call(page=1, cost=0.01)
            record_call(page=2, cost=0.02)
            log_summary()

        summary_records = [
            r for r in caplog.records
            if r.levelno == logging.INFO and "COST SUMMARY" in r.getMessage()
        ]
        assert len(summary_records) == 1
        msg = summary_records[0].getMessage()
        assert "total=$0.030000" in msg
        assert "across 2 LLM calls" in msg
        assert "2 pages" in msg

    def test_summary_no_calls_records_noop(self, caplog):
        with caplog.at_level(logging.INFO, logger="src.common.cost"):
            log_summary()

        summary_records = [
            r for r in caplog.records
            if "COST SUMMARY" in r.getMessage()
        ]
        assert len(summary_records) == 1
        assert "no LLM calls recorded" in summary_records[0].getMessage()


class TestThreadSafety:
    def test_concurrent_calls_produce_exact_total(self):
        n_threads = 16
        calls_per_thread = 50
        per_call_cost = 0.001

        def _worker():
            for _ in range(calls_per_thread):
                record_call(page=1, cost=per_call_cost)

        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(_worker) for _ in range(n_threads)]
            for f in as_completed(futures):
                f.result()

        expected_calls = n_threads * calls_per_thread
        expected_total = expected_calls * per_call_cost

        assert cost_module._totals.calls == expected_calls
        # Exact float arithmetic happens to hold for 0.001 * 800, but use
        # approx for robustness against FP jitter on different platforms.
        assert cost_module._totals.total_cost == pytest.approx(expected_total, rel=1e-9)
        # Only one page used, so by_page[1] == total.
        assert cost_module._totals.by_page[1] == pytest.approx(expected_total, rel=1e-9)