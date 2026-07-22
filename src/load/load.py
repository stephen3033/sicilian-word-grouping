"""Thread-safe incremental persistence for extracted dictionary pages."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from pathlib import Path
from threading import RLock
from typing import Any

from src.common.errors import ValidationError
from src.common.logger import log_errors
from src.config import Settings, get_settings
from src.models import DictionaryEntry

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2
_LEGACY_NAME_RE = re.compile(r"\.legacy(?:\.\d+)?\.json$")


class PersistenceCoordinator:
    """Own both output files and serialize every read-modify-write operation.

    The lock protects threads within one pipeline process. Each file is
    replaced atomically after its same-directory temporary file has been
    flushed and fsynced, so readers observe a complete old or new file.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        lock: RLock | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.dictionary_path = self.settings.dictionary_path(self.settings.model)
        self.failures_path = self.settings.failures_path()
        self._lock = lock or RLock()

        with self._lock:
            self.dictionary_path.parent.mkdir(parents=True, exist_ok=True)
            self.failures_path.parent.mkdir(parents=True, exist_ok=True)
            self._rename_pre_v2_outputs_locked()
            self._reconcile_failures_locked()

    def _empty_dictionary(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "volume": self.settings.volume,
            "model": self.settings.model,
            "pages": [],
            "page_count": 0,
            "entry_count": 0,
            "entries": [],
        }

    def _legacy_destination_locked(self, path: Path) -> Path:
        candidate = path.with_suffix(".legacy.json")
        sequence = 1
        while candidate.exists():
            candidate = path.with_suffix(f".legacy.{sequence}.json")
            sequence += 1
        return candidate

    def _rename_pre_v2_outputs_locked(self) -> None:
        """Move only old final volume envelopes out of the v2 namespace."""
        pattern = f"vs_{self.settings.volume}_*.json"
        for path in sorted(self.settings.output_dir.glob(pattern)):
            if _LEGACY_NAME_RE.search(path.name):
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValidationError(
                    f"cannot inspect existing final volume JSON {path}: {exc}"
                ) from exc

            version = payload.get("schema_version", 1) if isinstance(payload, dict) else 1
            if version == SCHEMA_VERSION:
                continue
            if not isinstance(version, int) or version > SCHEMA_VERSION:
                raise ValidationError(
                    f"unsupported schema_version {version!r} in {path}"
                )
            destination = self._legacy_destination_locked(path)
            path.replace(destination)
            logger.info("renamed pre-v2 output %s to %s", path, destination)

    def _read_dictionary_locked(self) -> dict[str, Any]:
        if not self.dictionary_path.exists():
            return self._empty_dictionary()
        try:
            payload = json.loads(self.dictionary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValidationError(
                f"cannot read dictionary JSON {self.dictionary_path}: {exc}"
            ) from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
            raise ValidationError(
                f"dictionary JSON {self.dictionary_path} is not a v{SCHEMA_VERSION} envelope"
            )
        if not isinstance(payload.get("pages"), list) or not isinstance(
            payload.get("entries"), list
        ):
            raise ValidationError(
                f"dictionary JSON {self.dictionary_path} has invalid pages or entries"
            )
        if any(not isinstance(page, int) for page in payload["pages"]):
            raise ValidationError(
                f"dictionary JSON {self.dictionary_path} has non-integer pages"
            )
        return payload

    def _read_failures_locked(self) -> set[int]:
        if not self.failures_path.exists():
            return set()
        failures: set[int] = set()
        for line_number, raw_line in enumerate(
            self.failures_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            value = raw_line.strip()
            if not value:
                continue
            try:
                page = int(value)
            except ValueError as exc:
                raise ValidationError(
                    f"invalid failure page on line {line_number}: {value!r}"
                ) from exc
            failures.add(page)
        return failures

    @staticmethod
    def _atomic_write_text(path: Path, text: str) -> None:
        """Flush and fsync a same-directory temp file, then atomically replace."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(text)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    def _write_dictionary_locked(self, payload: dict[str, Any]) -> None:
        self._atomic_write_text(
            self.dictionary_path,
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        )

    def _write_failures_locked(self, failures: set[int]) -> None:
        text = "".join(f"{page}\n" for page in sorted(failures))
        self._atomic_write_text(self.failures_path, text)

    def _reconcile_failures_locked(self) -> set[int]:
        pages = set(self._read_dictionary_locked()["pages"])
        failures = self._read_failures_locked()
        reconciled = failures - pages
        # Ensure the ledger exists even when it is empty.
        self._write_failures_locked(reconciled)
        if reconciled != failures:
            logger.info(
                "removed %d stale completed page(s) from %s",
                len(failures - reconciled),
                self.failures_path,
            )
        return reconciled

    @log_errors
    def reconcile_failures(self) -> set[int]:
        with self._lock:
            return self._reconcile_failures_locked()

    @log_errors
    def read_dictionary(self) -> dict[str, Any]:
        with self._lock:
            # Return an independent value so callers cannot mutate coordinator
            # state between locked operations.
            return json.loads(json.dumps(self._read_dictionary_locked()))

    @log_errors
    def read_failures(self) -> set[int]:
        with self._lock:
            return set(self._read_failures_locked())

    @log_errors
    def page_exists(self, page_number: int) -> bool:
        with self._lock:
            return page_number in self._read_dictionary_locked()["pages"]

    has_page = page_exists

    @log_errors
    def completed_pages(self) -> set[int]:
        with self._lock:
            return set(self._read_dictionary_locked()["pages"])

    @staticmethod
    def _entry_page(entry: dict[str, Any]) -> int:
        page_numbers = entry.get("page_numbers")
        if (
            not isinstance(page_numbers, list)
            or not page_numbers
            or not isinstance(page_numbers[0], int)
        ):
            raise ValidationError("persisted entry has invalid page_numbers")
        return page_numbers[0]

    @log_errors
    def insert_page(
        self, page_number: int, entries: list[DictionaryEntry]
    ) -> bool:
        """Atomically merge one complete page and clear its failure marker.

        Returns ``True`` when the page was inserted and ``False`` when another
        worker had already committed it. Failure cleanup happens under the
        same lock in both cases.
        """
        with self._lock:
            payload = self._read_dictionary_locked()
            pages = set(payload["pages"])
            inserted = page_number not in pages
            if inserted:
                new_entries = [entry.model_dump(mode="json") for entry in entries]
                if any(
                    entry.get("page_numbers") != [page_number]
                    for entry in new_entries
                ):
                    raise ValidationError(
                        f"page {page_number} insertion contains mismatched page_numbers"
                    )
                all_entries = [*payload["entries"], *new_entries]
                # Python's sort is stable, preserving source order within a page.
                all_entries.sort(key=self._entry_page)
                pages.add(page_number)
                ordered_pages = sorted(pages)
                payload.update(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "volume": self.settings.volume,
                        "model": self.settings.model,
                        "pages": ordered_pages,
                        "page_count": len(ordered_pages),
                        "entry_count": len(all_entries),
                        "entries": all_entries,
                    }
                )
                self._write_dictionary_locked(payload)

            failures = self._read_failures_locked()
            if page_number in failures:
                failures.remove(page_number)
                self._write_failures_locked(failures)
            logger.info(
                "page %d %s (%d entries) in %s",
                page_number,
                "inserted" if inserted else "already present",
                len(entries),
                self.dictionary_path,
            )
            return inserted

    persist_page = insert_page

    @log_errors
    def add_failure(self, page_number: int) -> None:
        with self._lock:
            completed = page_number in self._read_dictionary_locked()["pages"]
            failures = self._read_failures_locked()
            if completed and page_number in failures:
                failures.remove(page_number)
                self._write_failures_locked(failures)
            elif not completed and page_number not in failures:
                failures.add(page_number)
                self._write_failures_locked(failures)
            logger.info(
                "%s failed page %d in %s",
                "ignored completed" if completed else "recorded",
                page_number,
                self.failures_path,
            )

    record_failure = add_failure

    @log_errors
    def remove_failure(self, page_number: int) -> None:
        with self._lock:
            failures = self._read_failures_locked()
            if page_number in failures:
                failures.remove(page_number)
                self._write_failures_locked(failures)

    clear_failure = remove_failure
