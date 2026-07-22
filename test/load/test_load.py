"""Concurrency and atomicity tests for incremental persistence."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.config import Settings
from src.load import PersistenceCoordinator, SCHEMA_VERSION
from src.models import DictionaryEntry, ReviewStatus


def _entry(page: int, headword: str, *, orphan: bool = False) -> DictionaryEntry:
    return DictionaryEntry(
        headword=None if orphan else headword,
        trailing_text=f"body-{headword}",
        variants=None,
        page_numbers=[page],
        vs_vol=1,
        is_review_needed=(ReviewStatus.HUMAN if orphan else ReviewStatus.PASSED),
        review_reason=("orphan" if orphan else ""),
    )


def _coordinator(tmp_path: Path) -> PersistenceCoordinator:
    return PersistenceCoordinator(Settings(output_dir=tmp_path))


def test_insert_page_writes_v2_envelope_and_updates_metadata(tmp_path):
    persistence = _coordinator(tmp_path)
    persistence.insert_page(2, [_entry(2, "b"), _entry(2, "b2")])
    persistence.insert_page(1, [_entry(1, "a")])

    payload = json.loads(persistence.dictionary_path.read_text(encoding="utf-8"))
    assert payload == persistence.read_dictionary()
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["volume"] == 1
    assert payload["model"] == "anthropic/claude-sonnet-4.6"
    assert payload["pages"] == [1, 2]
    assert payload["page_count"] == 2
    assert payload["entry_count"] == 3
    assert [entry["headword"] for entry in payload["entries"]] == ["a", "b", "b2"]


def test_duplicate_page_is_rechecked_under_lock_and_not_inserted(tmp_path):
    persistence = _coordinator(tmp_path)
    assert persistence.insert_page(1, [_entry(1, "first")]) is True
    assert persistence.insert_page(1, [_entry(1, "duplicate")]) is False
    assert [item["headword"] for item in persistence.read_dictionary()["entries"]] == [
        "first"
    ]


def test_concurrent_out_of_order_insertions_have_no_loss_or_order_drift(tmp_path):
    persistence = _coordinator(tmp_path)

    def insert(page: int) -> None:
        persistence.insert_page(
            page,
            [_entry(page, f"{page}-first"), _entry(page, f"{page}-second")],
        )

    pages = list(range(80, 0, -1))
    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(insert, pages))

    payload = persistence.read_dictionary()
    assert payload["pages"] == list(range(1, 81))
    assert payload["page_count"] == 80
    assert payload["entry_count"] == 160
    assert [item["headword"] for item in payload["entries"]] == [
        value
        for page in range(1, 81)
        for value in (f"{page}-first", f"{page}-second")
    ]


def test_orphan_entry_is_preserved_as_its_own_source_order_record(tmp_path):
    persistence = _coordinator(tmp_path)
    persistence.insert_page(
        3, [_entry(3, "continuation", orphan=True), _entry(3, "next")]
    )
    entries = persistence.read_dictionary()["entries"]
    assert len(entries) == 2
    assert entries[0]["headword"] is None
    assert entries[0]["trailing_text"] == "body-continuation"
    assert entries[1]["headword"] == "next"


def test_concurrent_failure_updates_are_sorted_unique_and_lossless(tmp_path):
    persistence = _coordinator(tmp_path)
    additions = [9, 3, 7, 3, 5, 1, 9] * 20
    with ThreadPoolExecutor(max_workers=12) as executor:
        list(executor.map(persistence.add_failure, additions))
    assert persistence.read_failures() == {1, 3, 5, 7, 9}
    assert persistence.failures_path.read_text(encoding="utf-8") == "1\n3\n5\n7\n9\n"

    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(persistence.remove_failure, [3, 9, 3, 9]))
    assert persistence.failures_path.read_text(encoding="utf-8") == "1\n5\n7\n"


def test_success_and_failure_updates_converge_regardless_of_order(tmp_path):
    persistence = _coordinator(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(persistence.add_failure, 4),
            executor.submit(persistence.insert_page, 4, [_entry(4, "four")]),
        ]
        for future in futures:
            future.result()
    assert persistence.page_exists(4)
    assert 4 not in persistence.read_failures()


def test_startup_reconciles_stale_failures_and_keeps_empty_file(tmp_path):
    first = _coordinator(tmp_path)
    first.insert_page(2, [_entry(2, "two")])
    first.failures_path.write_text("1\n2\n", encoding="utf-8")

    second = _coordinator(tmp_path)
    assert second.read_failures() == {1}
    second.remove_failure(1)
    assert second.failures_path.exists()
    assert second.failures_path.read_text(encoding="utf-8") == ""


def test_pre_v2_final_is_renamed_collision_safely_without_touching_debug_files(
    tmp_path,
):
    settings = Settings(output_dir=tmp_path)
    final_path = settings.dictionary_path(settings.model)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.write_text(json.dumps({"volume": 1, "entries": []}), encoding="utf-8")
    final_path.with_suffix(".legacy.json").write_text("older", encoding="utf-8")
    debug_path = settings.validated_page_path(1, settings.model)
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path.write_text("debug", encoding="utf-8")

    persistence = PersistenceCoordinator(settings)
    assert not final_path.exists()
    assert final_path.with_suffix(".legacy.json").read_text() == "older"
    assert final_path.with_suffix(".legacy.1.json").exists()
    assert debug_path.read_text() == "debug"
    assert persistence.read_dictionary()["schema_version"] == 2


def test_atomic_replacement_never_exposes_partial_json_or_failure_lines(tmp_path):
    persistence = _coordinator(tmp_path)
    stop = threading.Event()
    errors: list[Exception] = []

    def observe() -> None:
        while not stop.is_set():
            try:
                if persistence.dictionary_path.exists():
                    json.loads(persistence.dictionary_path.read_text(encoding="utf-8"))
                if persistence.failures_path.exists():
                    for line in persistence.failures_path.read_text(encoding="utf-8").splitlines():
                        int(line)
            except Exception as exc:  # pragma: no cover - asserted below
                errors.append(exc)
                stop.set()

    observer = threading.Thread(target=observe)
    observer.start()
    try:
        for page in range(1, 35):
            persistence.add_failure(page)
            persistence.insert_page(page, [_entry(page, str(page))])
    finally:
        stop.set()
        observer.join(timeout=5)
    assert errors == []
