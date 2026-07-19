"""Unit tests for src.common.logger."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest

from src.common.errors import ValidationError
from src.common.logger import configure_logging, log_errors


@log_errors
def _boom():
    raise ValueError("kaboom")


@log_errors
def _ok():
    return 42


@log_errors
def _outer_calls_inner():
    _boom()


@log_errors
def _validation_fail():
    raise ValidationError("bad payload")


class TestLogErrorsDecorator:
    def test_decorator_logs_and_reraises(self, caplog):
        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValueError, match="kaboom"):
                _boom()

        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
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
        assert len(error_records) == 1
        assert "_boom failed" in error_records[0].getMessage()
        assert "_outer_calls_inner" not in error_records[0].getMessage()

    def test_decorator_preserves_function_name(self):
        assert _boom.__name__ == "_boom"
        assert _ok.__name__ == "_ok"

    def test_validation_error_logged_without_traceback(self, caplog):
        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValidationError, match="bad payload"):
                _validation_fail()

        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_records) == 1
        rec = error_records[0]
        assert "_validation_fail failed: bad payload" in rec.getMessage()
        assert rec.exc_info is None

    def test_validation_error_dedup_suppresses_outer_log(self, caplog):
        @log_errors
        def _outer():
            _validation_fail()

        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValidationError, match="bad payload"):
                _outer()

        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_records) == 1
        assert "_validation_fail failed" in error_records[0].getMessage()
        assert "_outer" not in error_records[0].getMessage()


class TestConfigureLogging:
    def test_quiet_library_loggers_pinned_to_error(self, tmp_path: Path):
        log_file = tmp_path / "pipeline.log"
        configure_logging(log_file)

        for name in ("openai", "httpx", "pydantic"):
            assert logging.getLogger(name).level == logging.ERROR

    def test_src_logger_at_debug(self, tmp_path: Path):
        log_file = tmp_path / "pipeline.log"
        configure_logging(log_file)
        assert logging.getLogger("src").level == logging.DEBUG

    def test_root_logger_at_warning(self, tmp_path: Path):
        log_file = tmp_path / "pipeline.log"
        configure_logging(log_file)
        assert logging.getLogger().level == logging.WARNING

    def test_idempotent_no_duplicate_handlers(self, tmp_path: Path):
        log_file = tmp_path / "pipeline.log"
        configure_logging(log_file)
        before = len(logging.getLogger().handlers)
        configure_logging(log_file)
        after = len(logging.getLogger().handlers)
        assert before == after == 1

    def test_format_includes_funcname(self, tmp_path: Path):
        log_file = tmp_path / "pipeline.log"
        configure_logging(log_file)
        logging.getLogger("src.test").debug("hello", stacklevel=2)
        for h in logging.getLogger().handlers:
            h.flush()
        text = log_file.read_text(encoding="utf-8")
        assert re.search(r"\] src\.test\.\w+: hello", text)
