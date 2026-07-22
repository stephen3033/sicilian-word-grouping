"""Single-pass orchestration, selectors, and work-conserving concurrency."""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

import pytest

from src.common.errors import ValidationError
from src.config import Settings
from src.load import PersistenceCoordinator
from src.main import (
    _build_parser,
    _process_page,
    _resolve_pages,
    _run_pages,
    main,
)
from src.models import DictionaryEntry, ReviewStatus


def _settings(tmp_path: Path, *, mode: str = "running") -> Settings:
    return Settings(
        mode=mode,
        output_dir=tmp_path / "out",
        raw_output_dir=tmp_path / "raw",
        log_file=tmp_path / "pipeline.log",
        track_cost=False,
    )


def _entry(page: int) -> DictionaryEntry:
    return DictionaryEntry(
        headword=f"page-{page}",
        trailing_text="body",
        variants=None,
        page_numbers=[page],
        vs_vol=1,
        is_review_needed=ReviewStatus.PASSED,
        review_reason="",
    )


def _parse(argv: list[str]) -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = _build_parser()
    return parser.parse_args(argv), parser


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["--start", "3", "--end", "5"], [3, 4, 5]),
        (["--pages", "9", "3", "9", "7"], [3, 7, 9]),
    ],
)
def test_range_and_inline_selectors_are_sorted_and_deduplicated(argv, expected):
    args, parser = _parse(argv)
    assert _resolve_pages(args, parser) == expected


def test_pages_file_selector_reads_one_page_per_line(tmp_path):
    page_file = tmp_path / "pages.txt"
    page_file.write_text("9\n3\n\n7\n3\n", encoding="utf-8")
    args, parser = _parse(["--pages-file", str(page_file)])
    assert _resolve_pages(args, parser) == [3, 7, 9]


@pytest.mark.parametrize(
    "argv",
    [
        ["--start", "3"],
        ["--pages", "3", "--end", "4"],
        ["--start", "3", "--end", "4", "--pages", "7"],
    ],
)
def test_page_selectors_are_mutually_exclusive_and_complete(argv):
    parser = _build_parser()
    with pytest.raises(SystemExit) as exc_info:
        args = parser.parse_args(argv)
        _resolve_pages(args, parser)
    assert exc_info.value.code == 2


def test_process_page_calls_each_layer_once_and_persists_immediately(
    monkeypatch, tmp_path
):
    settings = _settings(tmp_path)
    persistence = PersistenceCoordinator(settings)
    calls: list[str] = []

    def extract(page, active_settings):
        calls.append("extract")
        return "image", "ocr"

    def transform(image, ocr, page, active_settings):
        calls.append("transform")
        return '{"entries": []}'

    def validate_page(raw, ocr, image, page, active_settings):
        calls.append("validate")
        return [_entry(page)]

    monkeypatch.setattr("src.main._extract_page", extract)
    monkeypatch.setattr("src.main._transform_page", transform)
    monkeypatch.setattr("src.main._validate_page", validate_page)

    result = _process_page(5, settings, persistence)
    assert calls == ["extract", "transform", "validate"]
    assert result == [_entry(5)]
    assert persistence.page_exists(5)


def test_failed_page_is_recorded_once_without_retry(monkeypatch, tmp_path):
    settings = _settings(tmp_path)
    persistence = PersistenceCoordinator(settings)
    calls = 0

    monkeypatch.setattr("src.main._extract_page", lambda page, settings: ("img", "ocr"))

    def timeout(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise TimeoutError("request timed out")

    monkeypatch.setattr("src.main._transform_page", timeout)
    with pytest.raises(TimeoutError, match="timed out"):
        _process_page(2, settings, persistence)
    assert calls == 1
    assert persistence.read_failures() == {2}
    assert not persistence.page_exists(2)


def test_schema_failure_preserves_raw_response_and_excludes_page(
    monkeypatch, tmp_path
):
    settings = _settings(tmp_path)
    persistence = PersistenceCoordinator(settings)
    raw = "{malformed"
    monkeypatch.setattr("src.main._extract_page", lambda page, settings: ("img", "ocr"))
    monkeypatch.setattr("src.main._transform_page", lambda *args: raw)
    monkeypatch.setattr(
        "src.main._validate_page",
        lambda *args: (_ for _ in ()).throw(ValidationError("invalid envelope")),
    )

    with pytest.raises(ValidationError, match="invalid envelope"):
        _process_page(4, settings, persistence)
    assert settings.raw_page_path(4, settings.model).read_text() == raw
    assert persistence.read_failures() == {4}
    assert not persistence.dictionary_path.exists()


def test_fast_worker_persists_while_another_request_is_blocked(
    monkeypatch, tmp_path
):
    settings = _settings(tmp_path)
    persistence = PersistenceCoordinator(settings)
    blocked = threading.Event()
    release = threading.Event()
    fast_done = threading.Event()
    result: list[object] = []

    def fake_process(page, active_settings, active_persistence):
        if page == 1:
            blocked.set()
            assert release.wait(timeout=5)
        active_persistence.insert_page(page, [_entry(page)])
        if page == 2:
            fast_done.set()
        return [_entry(page)]

    monkeypatch.setattr("src.main._process_page", fake_process)

    runner = threading.Thread(
        target=lambda: result.append(_run_pages([1, 2], 2, settings, persistence))
    )
    runner.start()
    assert blocked.wait(timeout=2)
    assert fast_done.wait(timeout=2)
    assert persistence.page_exists(2)
    assert not persistence.page_exists(1)
    release.set()
    runner.join(timeout=5)
    assert result == [({1, 2}, {})]


def test_timeout_does_not_cancel_other_pages(monkeypatch, tmp_path):
    settings = _settings(tmp_path)
    persistence = PersistenceCoordinator(settings)

    def fake_process(page, active_settings, active_persistence):
        if page == 2:
            active_persistence.add_failure(page)
            raise TimeoutError("provider timeout")
        active_persistence.insert_page(page, [_entry(page)])
        return [_entry(page)]

    monkeypatch.setattr("src.main._process_page", fake_process)
    succeeded, failed = _run_pages([1, 2, 3, 4], 2, settings, persistence)
    assert succeeded == {1, 3, 4}
    assert set(failed) == {2}
    assert persistence.completed_pages() == {1, 3, 4}
    assert persistence.read_failures() == {2}


def test_main_skips_completed_pages_before_worker_submission(monkeypatch, tmp_path):
    settings = _settings(tmp_path)
    persistence = PersistenceCoordinator(settings)
    persistence.insert_page(3, [_entry(3)])
    submitted: list[int] = []

    monkeypatch.setattr("src.main.get_settings", lambda: settings)

    def fake_run(pages, batch_size, active_settings, active_persistence):
        submitted.extend(pages)
        return set(pages), {}

    monkeypatch.setattr("src.main._run_pages", fake_run)
    monkeypatch.setattr(sys, "argv", ["prog", "--pages", "7", "3", "7"])
    main()
    assert submitted == [7]


def test_main_exits_nonzero_after_all_requested_work_on_partial_failure(
    monkeypatch, tmp_path
):
    settings = _settings(tmp_path)
    monkeypatch.setattr("src.main.get_settings", lambda: settings)
    monkeypatch.setattr(
        "src.main._run_pages",
        lambda pages, batch, active_settings, persistence: (
            {1, 3},
            {2: TimeoutError("timeout")},
        ),
    )
    monkeypatch.setattr(sys, "argv", ["prog", "--start", "1", "--end", "3"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1


def test_batch_size_must_be_positive(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--pages", "1", "--batch-size", "0"],
    )
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2
