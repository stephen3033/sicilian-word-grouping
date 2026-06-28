# sicilian-word-grouping

Extract and group headwords from all variants of Sicilian dialect using the [Vocabolario Siciliano](https://it.wikipedia.org/wiki/Vocabolario_Siciliano) PDF volumes and OpenAI-compatible vision-language models (Ollama, OpenRouter, etc.).

## Architecture Overview

The system is split into four isolated, data-driven layers:

1. **Extract (E):** Handles raw file ingestion. It splits document pages into high-resolution visual layouts (Base64 images) and extracts character-accurate token data from the underlying text layer concurrently.
2. **Transform (T):** Maps layout data and token arrays into structured model payloads via a multimodal OpenRouter vision call (one composited page image + one compiled prompt per page). `src/main.py` orchestrates a configurable `--start`/`--end` printed-page range, running E -> T per page and persisting each page's raw model output to `test/data/transform/output/` for visual inspection.
3. **Validate (V):** Instantiates a runtime quality gate. It infers reference-free metrics (like schema syntax compliance and OCR semantic grounding scores) to catch hallucinations on the fly without needing pre-existing golden datasets.
4. **Load (L):** Manages structural persistence, logging clean extracted JSON outputs alongside their analytical evaluation metrics.

---

## Extraction Layer

The two extractors share a common addressing scheme: the integer argument is the **printed page number** of the active volume (selected by `VS_VOLUME`). The Vocabolario Siciliano single-column PDFs split each printed page into two physical PDF pages (left column, then right column), so one printed page = two PDF pages. `extract_page_image` renders both and composites them into a single image; `extract_page_text` returns the matching block from the OCR txt (one line per OCR'd line, prefixed `<n> <text>`).

- **Image composition** (`VS_COLUMN_LAYOUT`): `vertical` (default) stacks the left column above the right, so a VLM's natural top-down scan matches the dictionary's reading order (left col top→bottom, then right col top→bottom). `horizontal` restores the original side-by-side page orientation for A/B testing.
- **Secrets**: `.env` holds only `OPENAI_API_KEY=op://...` and is resolved by 1Password — launch with `op run --env-file=.env -- uv run sicilian-word-grouping --start 1 --end 2`. `Settings` (`src/config.py`) never reads `.env`; it reads the resolved values from the process environment. All other config (`VS_VOLUME`, `VS_DPI`, `VS_COLUMN_LAYOUT`, `VS_DATA_DIR`, `VS_STRIP_OCR_PREFIX`, `OPENAI_BASE_URL`, `MODEL`) has Python defaults and is overridable via env.

---

## Testing

Run the unit suite (no network or API key required; the OpenAI client is faked):

```bash
uv run pytest
```