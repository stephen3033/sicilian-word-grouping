"""Unit tests for src.main (orchestrator).

Covers retry orchestration, fatal-failure handling, and the per-page
E -> T -> V flow. Tests monkeypatch the per-layer helpers
(``_extract_page``, ``_transform_page``, ``_validate_page``) and
``stitch`` so the orchestrator is exercised without touching the
network, filesystem, or PDF.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from src.common.errors import ValidationError
from src.config import Settings
from src.main import _process_page_with_retries, main
from src.models import DictionaryEntry


def _entry(headword: str = "a²") -> DictionaryEntry:
    return DictionaryEntry.model_construct(
        headword=headword,
        trailing_text="art. femm. la.",
        variants=None,
        page_numbers=[1],
        vs_vol=1,
    )


class _Recorder:
    """Tracks per-layer calls for assertions."""

    def __init__(self) -> None:
        self.extract_calls: list[int] = []
        self.transform_calls: list[tuple[int, int]] = []  # (page, attempt)
        self.validate_calls: list[tuple[int, int]] = []  # (page, attempt)
        self.validate_results: dict[int, list[DictionaryEntry]] = {}
        self.transform_payloads: dict[int, list[str]] = {}


def _make_transform_payload(attempt: int) -> str:
    return f'{{"entries": [{{"headword": "attempt{attempt}"}}]}}'


def _patch_layers(monkeypatch, recorder: _Recorder, settings: Settings):
    """Replace src.main per-layer helpers with recorder-aware stubs."""

    def _fake_extract(page, settings):
        recorder.extract_calls.append(page)
        return ("img-b64", "ocr-text")

    def _fake_transform(image_b64, ocr_text, page, settings, attempt=1):
        recorder.transform_calls.append((page, attempt))
        payload = _make_transform_payload(attempt)
        recorder.transform_payloads.setdefault(page, []).append(payload)
        # Mirror the real persistence rules so tests can assert on-disk artifacts.
        if attempt == 1:
            if settings.mode == "debug":
                out = settings.raw_page_path(page, settings.model)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(payload, encoding="utf-8")
        else:
            out = settings.raw_retry_page_path(page, settings.model, attempt)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(payload, encoding="utf-8")
        return payload

    def _make_validate(fail_attempts: set[int]):
        def _fake_validate(raw_json, ocr_text, image_b64, page, settings):
            attempt = len(recorder.validate_calls) + 1
            recorder.validate_calls.append((page, attempt))
            if attempt in fail_attempts:
                raise ValidationError(f"attempt {attempt} bad")
            return [_entry()]
        return _fake_validate

    return _fake_extract, _fake_transform, _make_validate


def _settings(tmp_path: Path, mode: str = "running") -> Settings:
    return Settings(
        mode=mode,
        raw_output_dir=tmp_path / "raw",
        output_dir=tmp_path / "out",
        log_file=tmp_path / "pipeline.log",
    )


class TestProcessPageSuccess:
    def test_first_attempt_success_calls_each_layer_once(
        self, monkeypatch, tmp_path
    ):
        s = _settings(tmp_path)
        rec = _Recorder()
        ext, tr, mk_val = _patch_layers(monkeypatch, rec, s)
        monkeypatch.setattr("src.main._extract_page", ext)
        monkeypatch.setattr("src.main._transform_page", tr)
        monkeypatch.setattr("src.main._validate_page", mk_val(set()))

        out = _process_page_with_retries(5, s)

        assert rec.extract_calls == [5]
        assert rec.transform_calls == [(5, 1)]
        assert rec.validate_calls == [(5, 1)]
        assert len(out) == 1
        # No retry files written in running mode on first-attempt success.
        assert not s.raw_page_path(5, s.model).exists()
        assert not s.raw_retry_page_path(5, s.model, 2).exists()


class TestProcessPageRetries:
    def test_second_attempt_success_writes_both_artifacts_in_running_mode(
        self, monkeypatch, tmp_path
    ):
        s = _settings(tmp_path, mode="running")
        rec = _Recorder()
        ext, tr, mk_val = _patch_layers(monkeypatch, rec, s)
        monkeypatch.setattr("src.main._extract_page", ext)
        monkeypatch.setattr("src.main._transform_page", tr)
        # First attempt fails validation, second succeeds.
        monkeypatch.setattr("src.main._validate_page", mk_val({1}))

        out = _process_page_with_retries(5, s)

        assert rec.extract_calls == [5]  # extracted once even across retries
        assert rec.transform_calls == [(5, 1), (5, 2)]
        assert rec.validate_calls == [(5, 1), (5, 2)]
        assert len(out) == 1
        # Attempt 1 failed in running mode -> orchestrator persisted raw_page_path.
        assert s.raw_page_path(5, s.model).exists()
        assert (
            s.raw_page_path(5, s.model).read_text(encoding="utf-8")
            == _make_transform_payload(1)
        )
        # Attempt 2 (retry) -> _transform_page persisted retry path.
        assert s.raw_retry_page_path(5, s.model, 2).exists()
        assert (
            s.raw_retry_page_path(5, s.model, 2).read_text(encoding="utf-8")
            == _make_transform_payload(2)
        )

    def test_third_attempt_success_writes_retry2_and_retry3(
        self, monkeypatch, tmp_path
    ):
        s = _settings(tmp_path, mode="running")
        rec = _Recorder()
        ext, tr, mk_val = _patch_layers(monkeypatch, rec, s)
        monkeypatch.setattr("src.main._extract_page", ext)
        monkeypatch.setattr("src.main._transform_page", tr)
        monkeypatch.setattr("src.main._validate_page", mk_val({1, 2}))

        out = _process_page_with_retries(7, s)

        assert rec.transform_calls == [(7, 1), (7, 2), (7, 3)]
        assert rec.validate_calls == [(7, 1), (7, 2), (7, 3)]
        assert len(out) == 1
        assert s.raw_page_path(7, s.model).exists()
        assert s.raw_retry_page_path(7, s.model, 2).exists()
        assert s.raw_retry_page_path(7, s.model, 3).exists()

    def test_all_attempts_fail_raises_validation_error(
        self, monkeypatch, tmp_path
    ):
        s = _settings(tmp_path, mode="running")
        rec = _Recorder()
        ext, tr, mk_val = _patch_layers(monkeypatch, rec, s)
        monkeypatch.setattr("src.main._extract_page", ext)
        monkeypatch.setattr("src.main._transform_page", tr)
        monkeypatch.setattr("src.main._validate_page", mk_val({1, 2, 3}))

        with pytest.raises(ValidationError, match="failed after 3 attempts"):
            _process_page_with_retries(5, s)

        assert rec.transform_calls == [(5, 1), (5, 2), (5, 3)]
        assert rec.validate_calls == [(5, 1), (5, 2), (5, 3)]
        # All three attempt artifacts on disk.
        assert s.raw_page_path(5, s.model).exists()
        assert s.raw_retry_page_path(5, s.model, 2).exists()
        assert s.raw_retry_page_path(5, s.model, 3).exists()

    def test_non_validation_error_in_transform_is_not_retried(
        self, monkeypatch, tmp_path
    ):
        s = _settings(tmp_path)
        rec = _Recorder()

        def _boom(image, ocr, page, settings, attempt=1):
            rec.transform_calls.append((page, attempt))
            raise RuntimeError("network down")

        monkeypatch.setattr("src.main._extract_page",
                            _patch_layers(monkeypatch, rec, s)[0])
        monkeypatch.setattr("src.main._transform_page", _boom)

        with pytest.raises(RuntimeError, match="network down"):
            _process_page_with_retries(5, s)

        # Only one transform call; no retry on non-ValidationError.
        assert rec.transform_calls == [(5, 1)]

    def test_validation_error_in_transform_is_retried(
        self, monkeypatch, tmp_path
    ):
        # e.g. the client's empty-choices guard: a transient provider glitch
        # raises ValidationError from the transform layer and must retry.
        s = _settings(tmp_path)
        rec = _Recorder()
        ext, tr, mk_val = _patch_layers(monkeypatch, rec, s)

        def _flaky_transform(image, ocr, page, settings, attempt=1):
            if attempt == 1:
                rec.transform_calls.append((page, attempt))
                raise ValidationError("model returned no choices")
            return tr(image, ocr, page, settings, attempt=attempt)

        monkeypatch.setattr("src.main._extract_page", ext)
        monkeypatch.setattr("src.main._transform_page", _flaky_transform)
        monkeypatch.setattr("src.main._validate_page", mk_val(set()))

        out = _process_page_with_retries(5, s)

        assert rec.transform_calls == [(5, 1), (5, 2)]
        assert len(out) == 1
        # No raw_json existed for the failed attempt, so nothing was persisted
        # to raw_page_path.
        assert not s.raw_page_path(5, s.model).exists()


class TestCostSummaryGuarantee:
    """The COST SUMMARY line must emit from main()'s finally block
    whether the pipeline succeeds, a page fails fatally, or stitch raises."""

    def _patch_pipeline(self, monkeypatch, tmp_path, process_page_fn, *, stitch_raises=False):
        s = _settings(tmp_path, mode="running")
        monkeypatch.setattr("src.main.get_settings", lambda: s)
        monkeypatch.setattr("src.main._process_page_with_retries", process_page_fn)

        if stitch_raises:
            def _boom_stitch(entries_by_page, settings=None):
                raise RuntimeError("stitch blew up")
            monkeypatch.setattr("src.main.stitch", _boom_stitch)
        else:
            monkeypatch.setattr(
                "src.main.stitch",
                lambda entries_by_page, settings=None: tmp_path / "stitched.json",
            )

        calls: list[int] = []

        def _spy_log_summary():
            calls.append(1)

        monkeypatch.setattr("src.main.log_summary", _spy_log_summary)
        return s, calls

    def test_summary_emitted_on_success(self, monkeypatch, tmp_path):
        _, calls = self._patch_pipeline(
            monkeypatch,
            tmp_path,
            lambda page, settings: [_entry(headword=f"p{page}")],
        )
        monkeypatch.setattr(sys, "argv", ["prog", "--start", "1", "--end", "3"])

        main()

        assert calls == [1]

    def test_summary_emitted_on_fatal_page_failure(self, monkeypatch, tmp_path):
        def _fake(page, settings):
            if page == 2:
                raise ValidationError("page 2 unrecoverable")
            return [_entry(headword=f"p{page}")]

        _, calls = self._patch_pipeline(monkeypatch, tmp_path, _fake)
        monkeypatch.setattr(sys, "argv", ["prog", "--start", "1", "--end", "3"])

        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 1
        # The finally block ran log_summary *before* SystemExit propagated.
        assert calls == [1]

    def test_summary_emitted_when_stitch_raises(self, monkeypatch, tmp_path):
        _, calls = self._patch_pipeline(
            monkeypatch,
            tmp_path,
            lambda page, settings: [_entry(headword=f"p{page}")],
            stitch_raises=True,
        )
        monkeypatch.setattr(sys, "argv", ["prog", "--start", "1", "--end", "2"])

        with pytest.raises(RuntimeError, match="stitch blew up"):
            main()
        assert calls == [1]

    def test_summary_skipped_when_track_cost_false(self, monkeypatch, tmp_path):
        s, calls = self._patch_pipeline(
            monkeypatch,
            tmp_path,
            lambda page, settings: [_entry(headword=f"p{page}")],
        )
        s.track_cost = False
        monkeypatch.setattr("src.main.get_settings", lambda: s)
        monkeypatch.setattr(sys, "argv", ["prog", "--start", "1", "--end", "3"])

        main()

        assert calls == []


class TestMainFlow:
    def test_fatal_page_stops_loop_and_stitches_succeeded_pages(
        self, monkeypatch, tmp_path
    ):
        s = _settings(tmp_path, mode="running")
        stitched_arg: dict = {}

        def _fake_process_page(page, settings):
            if page == 2:
                raise ValidationError("page 2 unrecoverable")
            return [_entry(headword=f"p{page}")]

        def _fake_stitch(entries_by_page, settings=None):
            stitched_arg["pages"] = sorted(entries_by_page)
            stitched_arg["entries_by_page"] = entries_by_page
            return tmp_path / "stitched.json"

        monkeypatch.setattr("src.main.get_settings", lambda: s)
        monkeypatch.setattr("src.main._process_page_with_retries", _fake_process_page)
        monkeypatch.setattr("src.main.stitch", _fake_stitch)

        argv = ["prog", "--start", "1", "--end", "3"]
        monkeypatch.setattr(sys, "argv", argv)

        with pytest.raises(SystemExit) as excinfo:
            main()

        assert excinfo.value.code == 1
        # Only page 1 succeeded before fatal page 2 stopped the loop.
        assert stitched_arg["pages"] == [1]
        assert stitched_arg["entries_by_page"][1][0].headword == "p1"

    def test_all_pages_succeed_exits_zero(self, monkeypatch, tmp_path):
        s = _settings(tmp_path, mode="running")
        stitched_arg: dict = {}

        def _fake_process_page(page, settings):
            return [_entry(headword=f"p{page}")]

        def _fake_stitch(entries_by_page, settings=None):
            stitched_arg["pages"] = sorted(entries_by_page)
            return tmp_path / "stitched.json"

        monkeypatch.setattr("src.main.get_settings", lambda: s)
        monkeypatch.setattr("src.main._process_page_with_retries", _fake_process_page)
        monkeypatch.setattr("src.main.stitch", _fake_stitch)
        monkeypatch.setattr(sys, "argv", ["prog", "--start", "1", "--end", "3"])

        main()  # should not raise SystemExit
        assert stitched_arg["pages"] == [1, 2, 3]


class TestMainParallelFlow:
    """Tests for the --batch-size parallel path in src.main.main."""

    def test_batch_size_exceeds_page_count_errors(self, monkeypatch, tmp_path):
        s = _settings(tmp_path, mode="running")
        monkeypatch.setattr("src.main.get_settings", lambda: s)
        monkeypatch.setattr("src.main._process_page_with_retries",
                            lambda page, settings: [_entry()])
        monkeypatch.setattr("src.main.stitch",
                            lambda entries_by_page, settings=None: tmp_path / "x.json")
        monkeypatch.setattr(sys, "argv",
                            ["prog", "--start", "1", "--end", "3", "--batch-size", "10"])

        with pytest.raises(SystemExit) as excinfo:
            main()
        # argparse.parser.error exits with status 2.
        assert excinfo.value.code == 2

    def test_batch_size_zero_errors(self, monkeypatch, tmp_path):
        s = _settings(tmp_path, mode="running")
        monkeypatch.setattr("src.main.get_settings", lambda: s)
        monkeypatch.setattr("src.main._process_page_with_retries",
                            lambda page, settings: [_entry()])
        monkeypatch.setattr("src.main.stitch",
                            lambda entries_by_page, settings=None: tmp_path / "x.json")
        monkeypatch.setattr(sys, "argv",
                            ["prog", "--start", "1", "--end", "3", "--batch-size", "0"])

        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 2

    def test_batch_size_negative_errors(self, monkeypatch, tmp_path):
        s = _settings(tmp_path, mode="running")
        monkeypatch.setattr("src.main.get_settings", lambda: s)
        monkeypatch.setattr("src.main._process_page_with_retries",
                            lambda page, settings: [_entry()])
        monkeypatch.setattr("src.main.stitch",
                            lambda entries_by_page, settings=None: tmp_path / "x.json")
        monkeypatch.setattr(sys, "argv",
                            ["prog", "--start", "1", "--end", "3",
                             "--batch-size", "-1"])

        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 2

    def test_batch_size_equals_page_count_ok(self, monkeypatch, tmp_path):
        s = _settings(tmp_path, mode="running")
        stitched_arg: dict = {}

        def _fake_process_page(page, settings):
            return [_entry(headword=f"p{page}")]

        def _fake_stitch(entries_by_page, settings=None):
            stitched_arg["pages"] = sorted(entries_by_page)
            return tmp_path / "stitched.json"

        monkeypatch.setattr("src.main.get_settings", lambda: s)
        monkeypatch.setattr("src.main._process_page_with_retries", _fake_process_page)
        monkeypatch.setattr("src.main.stitch", _fake_stitch)
        monkeypatch.setattr(sys, "argv",
                            ["prog", "--start", "1", "--end", "3",
                             "--batch-size", "3"])

        main()  # no SystemExit
        assert stitched_arg["pages"] == [1, 2, 3]

    def test_all_pages_succeed_in_parallel(self, monkeypatch, tmp_path):
        s = _settings(tmp_path, mode="running")
        stitched_arg: dict = {}

        def _fake_process_page(page, settings):
            return [_entry(headword=f"p{page}")]

        def _fake_stitch(entries_by_page, settings=None):
            stitched_arg["pages"] = sorted(entries_by_page)
            return tmp_path / "stitched.json"

        monkeypatch.setattr("src.main.get_settings", lambda: s)
        monkeypatch.setattr("src.main._process_page_with_retries", _fake_process_page)
        monkeypatch.setattr("src.main.stitch", _fake_stitch)
        monkeypatch.setattr(sys, "argv",
                            ["prog", "--start", "1", "--end", "4",
                             "--batch-size", "2"])

        main()
        assert stitched_arg["pages"] == [1, 2, 3, 4]

    def test_fatal_page_completes_in_flight_then_stitches_and_exits_nonzero(
        self, monkeypatch, tmp_path
    ):
        s = _settings(tmp_path, mode="running")
        stitched_arg: dict = {}

        def _fake_process_page(page, settings):
            if page == 2:
                raise ValidationError("page 2 unrecoverable")
            return [_entry(headword=f"p{page}")]

        def _fake_stitch(entries_by_page, settings=None):
            stitched_arg["pages"] = sorted(entries_by_page)
            stitched_arg["entries_by_page"] = entries_by_page
            return tmp_path / "stitched.json"

        monkeypatch.setattr("src.main.get_settings", lambda: s)
        monkeypatch.setattr("src.main._process_page_with_retries", _fake_process_page)
        monkeypatch.setattr("src.main.stitch", _fake_stitch)
        monkeypatch.setattr(sys, "argv",
                            ["prog", "--start", "1", "--end", "4",
                             "--batch-size", "2"])

        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 1
        # Page 2 itself failed; at least one non-fatal page should have
        # been collected (page 1 ran concurrently). Pages 3/4 may or may
        # not have started depending on scheduling.
        assert 2 not in stitched_arg["pages"]
        assert 1 in stitched_arg["pages"]