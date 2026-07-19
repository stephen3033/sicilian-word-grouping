"""Load layer: stitch validated per-page entries into a single volume JSON.

In both ``debug`` and ``running`` modes the load layer is the only layer
guaranteed to write to disk. ``stitch`` takes a page-keyed dict of
already-validated entries (held in memory by the orchestrator) and emits
one stitched volume file at ``settings.stitched_path(model)`` with a
metadata envelope (``volume``, ``model``, ``page_count``,
``entry_count``) and a flat ``entries`` list in page order.

``read_pages_from_disk`` is exposed for standalone load re-runs (e.g. a
debug-mode resumability workflow) where the orchestrator is skipped and
the load layer reads back the per-page JSON previously written by
``src.validate.persist_validated_page``.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from src.common.logger import log_errors
from src.config import Settings, get_settings
from src.models import DictionaryEntry

logger = logging.getLogger(__name__)

_PAGE_RE = re.compile(r"VS\d+_page_(\d+)_.*\.json$")


@log_errors
def stitch(
    entries_by_page: dict[int, list[DictionaryEntry]],
    settings: Settings | None = None,
) -> Path:
    """Concatenate validated entries (in page order) into one volume JSON.

    Returns the path written. Parents are created as needed.
    """
    if settings is None:
        settings = get_settings()

    ordered_pages = sorted(entries_by_page)
    all_entries: list[DictionaryEntry] = []
    for page in ordered_pages:
        all_entries.extend(entries_by_page[page])

    out_path = settings.stitched_path(settings.model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "volume": settings.volume,
        "model": settings.model,
        "page_count": len(ordered_pages),
        "entry_count": len(all_entries),
        "entries": [e.model_dump() for e in all_entries],
    }
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        "stitched %d pages / %d entries to %s",
        len(ordered_pages),
        len(all_entries),
        out_path,
    )
    return out_path


@log_errors
def read_pages_from_disk(
    settings: Settings | None = None,
) -> dict[int, list[DictionaryEntry]]:
    """Read all validated per-page JSON from disk into a page-keyed dict.

    Intended for standalone load re-runs (debug mode resumability). The
    orchestrator does not call this: it passes its in-memory
    ``entries_by_page`` directly to ``stitch``.
    """
    if settings is None:
        settings = get_settings()

    pages_dir = settings.validated_pages_dir()
    result: dict[int, list[DictionaryEntry]] = {}
    if not pages_dir.exists():
        logger.warning("validated pages dir not found: %s", pages_dir)
        return result

    for path in sorted(pages_dir.glob("*.json")):
        m = _PAGE_RE.search(path.name)
        if not m:
            logger.debug("skipping unrecognized filename: %s", path.name)
            continue
        page_number = int(m.group(1))
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_entries = payload.get("entries", [])
        # The persisted JSON was validated at write time (grounding, schema,
        # layout), so we bypass re-validation here via ``model_construct``;
        # the OCR text needed for grounding isn't available at load time.
        result[page_number] = [
            DictionaryEntry.model_construct(**e) for e in raw_entries
        ]
        logger.debug("read page %d (%d entries) from %s",
                      page_number, len(result[page_number]), path)
    logger.info("read %d pages from %s", len(result), pages_dir)
    return result