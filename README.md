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

## Transform Layer

The transform layer maps each page's composited image and OCR text into a structured `DictionaryEntry` payload via a single multimodal OpenRouter vision call (one image + one compiled prompt per page).

- **Prompt compilation** (`src/transform/prompter.py`): `build_user_prompt` selects a model-specific preamble from `_USER_PREAMBLES` keyed by the active `MODEL`, then concatenates it with the `DictionaryEntry` JSON schema and the OCR page text via `_USER_TEMPLATE`. The model-agnostic `SYSTEM_PROMPT` frames the call as a non-conversational extraction engine.
- **Vision client** (`src/transform/client.py`): `extract_json` sends the system prompt, the Base64 page image, and the compiled user prompt to the OpenAI-compatible chat endpoint. `response_format={"type": "json_object"}` is a hard rail forcing raw JSON output (no markdown wrappers).
- **Supported models** (selectable via the `MODEL` env var): `anthropic/claude-sonnet-4.6` (default) and `google/gemini-3.1-pro-preview`. Each has its own preamble string in `_USER_PREAMBLES` so they can be refined independently; the Gemini preamble currently mirrors the Claude preamble as a shared starting point. Note: `gemini-3.1-pro-preview` is an OpenRouter preview slug and may be renamed or retired upstream.
- **Persistence**: `src/main.py` orchestrates a configurable `--start`/`--end` printed-page range, running E -> T per page and writing each page's raw model output to `test/data/transform/output/` for visual inspection.

---

## Testing

Run the unit suite (no network or API key required; the OpenAI client is faked):

```bash
uv run pytest
```