"""Unit tests for src.common.logger.log_errors decorator."""

from __future__ import annotations

import logging
import re

import pytest

from src.common.logger import log_errors


@log_errors
def _boom():
    raise ValueError("kaboom")


@log_errors
def _ok():
    return 42


@log_errors
def _outer_calls_inner():
    _boom()


def _record_records(caplog):
    """Return list of LogRecord captured by caplog at any level."""
    return caplog.records


class TestLogErrorsDecorator:
    def test_decorator_logs_and_reraises(self, caplog):
        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValueError, match="kaboom"):
                _boom()

        records = _record_records(caplog)
        error_records = [r for r in records if r.levelno == logging.ERROR]
        assert len(error_records) == 1
        rec = error_records[0]
        assert "_boom failed" in rec.getMessage()
        assert rec.exc_info is not None
        assert rec.exc_info[0] is ValueError

    def test_decorator_success_passthrough(self, caplog):
        with caplog.at_level(logging.DEBUG):
            assert _ok() == 42

        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert error_records == []

    def test_decorator_dedup_suppresses_outer_log(self, caplog):
        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValueError, match="kaboom"):
                _outer_calls_inner()

        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        # Only the inner _boom logs; _outer_calls_inner skips re-logging.
        assert len(error_records) == 1
        assert "_boom failed" in error_records[0].getMessage()
        assert "_outer_calls_inner" not in error_records[0].getMessage()

    def test_decorator_preserves_function_name(self):
        assert _boom.__name__ == "_boom"
        assert _ok.__name__ == "_ok"
