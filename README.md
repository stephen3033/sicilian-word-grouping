# sicilian-word-grouping

Extract and group headwords from all variants of Sicilian dialect using the [Vocabolario Siciliano](https://it.wikipedia.org/wiki/Vocabolario_Siciliano) PDF volumes and OpenAI-compatible vision-language models (Ollama, OpenRouter, etc.).

## Architecture Overview

The system is split into four isolated, data-driven layers:

1. **Extract (E):** Renders a printed page of the active VS volume to a Base64 PNG (compositing its two physical PDF pages) and reads the matching OCR text block from a pre-rendered txt file.
2. **Transform (T):** Compiles a single prompt (preamble + `DictionaryEntry` JSON schema + OCR text) and sends it with the page image to an OpenAI-compatible VLM; returns raw JSON. `src/main.py` orchestrates a configurable `--start`/`--end` printed-page range, running E -> T per page (persisting raw JSON in debug mode, or on first-attempt validation failure in running mode).
3. **Validate (V):** Unwraps `{"entries": [...]}`, checks Pydantic schema conformance, grounds `headword`/`variants`/`trailing_text` against the OCR text, and applies a pixel-based first-entry layout heuristic — all without a golden dataset.
4. **Load (L):** Stitches validated per-page entries into a single volume JSON with a metadata envelope (`volume`, `model`, `page_count`, `entry_count`).

---

## Extract Layer

The two extractors share a common addressing scheme: the integer argument is the **printed page number** of the active volume (selected by `VS_VOLUME`). VS single-column PDFs split each printed page into two physical PDF pages (left column, then right column), so one printed page = two PDF pages. `extract_page_image` renders both and composites them; `extract_page_text` returns the matching block from the OCR txt (one line per OCR'd line, prefixed `<n> <text>`).

- **Image composition** (`VS_COLUMN_LAYOUT`): `vertical` (default) stacks the left column above the right so a VLM's top-down scan matches reading order; `horizontal` restores side-by-side for A/B testing.
- **OCR indexing** (`extract_page_text`): builds a `{printed_page: [lines]}` index from the txt once (mtime-keyed cache), strips the `<n> ` prefix when `VS_STRIP_OCR_PREFIX` is true.

---

## Transform Layer

- **Prompt compilation** (`src/transform/prompter.py`): `build_user_prompt` concatenates the single model-agnostic `DEFAULT_USER_PREAMBLE`, the `DictionaryEntry` JSON schema, and the OCR page text via `_USER_TEMPLATE`. `SYSTEM_PROMPT` frames the call as a non-conversational extraction engine.
- **Vision client** (`src/transform/client.py`): `extract_json` sends system + image + user prompts to the OpenAI-compatible chat endpoint with `response_format={"type": "json_object"}` (hard rail, no markdown wrappers). `page` is keyword-only so per-call cost is attributed to the right page. Default `MODEL` is `anthropic/claude-sonnet-4.6`; any OpenRouter-compatible model id works.
- **Persistence**: attempt-1 raw JSON written only in debug mode; retry-attempt raw JSON always written (`_retry<N>.json`); first-attempt raw JSON preserved on validation failure even in running mode.

---

## Validate Layer

- **Schema** (`src/validate/validate.py`): unwraps `{"entries": [...]}` and validates each entry via Pydantic `model_validator(mode="after")`.
- **Grounding** (`src/common/normalize.py`): NFC + whitespace-normalized substring match against OCR text for `headword`, `variants`, `trailing_text`.
- **Layout heuristic**: on the first entry only, compares the pixel-bbox left-X of the first two real text lines; `|Δx| > headword_delta - tolerance` ⇒ headword expected present, otherwise `None` (orphan).
- **Injection**: `vs_vol` and `page_numbers` (overriding the model's `0` / `[0]` placeholders) are set in the same per-entry pass.
- **Persistence** (`persist_validated_page`): writes per-page validated JSON (debug mode only).

---

## Load Layer

- **`stitch`** (`src/load/load.py`): the only layer that always writes to disk (both modes). Emits `vs_<vol>_<model>.json` with a metadata envelope (`volume`, `model`, `page_count`, `entry_count`) and a flat `entries` list in page order.
- **`read_pages_from_disk`**: standalone re-run entry point for debug-mode resumability; reads per-page JSON previously written by `persist_validated_page` and bypasses re-validation via `model_construct` (grounding OCR isn't available at load time).

---

## Usage

```bash
op run --env-file=.env -- uv run sicilian-word-grouping --start <n> --end <m> [--mode debug|running] [--batch-size <n>]
```

### Flags

- `--start` / `--end` — first and last printed page (inclusive).
- `--mode` — `debug` persists per-page artifacts from transform and validate; `running` keeps validated entries in memory (load still writes). Overrides `VS_MODE`.
- `--batch-size` — parallelize the T→V loop across this many pages at once. Sequential if omitted. Must be ≤ total pages in range.

### Secrets

`.env` holds only `OPENAI_API_KEY=op://...` and is resolved by 1Password at launch via `op run --env-file=.env -- ...`. `Settings` (`src/config.py`) never reads `.env`; it reads the resolved values from the process environment. See `.env.template` (`OPENAI_API_KEY=sk-...`).

### Environment Variables

| Variable | Default | Notes |
| --- | --- | --- |
| `OPENAI_API_KEY` | — | Required. Resolved by 1Password. |
| `OPENAI_BASE_URL` | `https://openrouter.ai/api/v1` | OpenAI-compatible endpoint. |
| `MODEL` | `anthropic/claude-sonnet-4.6` | Any OpenRouter-compatible model id. |
| `VS_VOLUME` | `1` | Active VS volume number. |
| `VS_DATA_DIR` | `VS` | Root for PDF (`columns/`) and OCR (`OCR_cols/`). |
| `VS_DPI` | `200` | PDF render resolution. |
| `VS_COLUMN_LAYOUT` | `vertical` | `vertical` or `horizontal`. |
| `VS_STRIP_OCR_PREFIX` | `true` | Strip `<n> ` line prefix from OCR txt. |
| `VS_MODE` | `running` | `debug` or `running`. |
| `VS_MAX_ATTEMPTS` | `3` | T→V retries per page before fatal. |
| `VS_TRACK_COST` | `true` | Track per-call OpenRouter cost. |
| `VS_HEADWORD_DELTA` | `36.0` | Calibrated min \|Δx\| px for layout heuristic. |
| `VS_LAYOUT_TOLERANCE` | `15.0` | Px subtracted from `headword_delta`. |
| `VS_RAW_OUTPUT_DIR` | `test/data/transform/output` | Transform raw JSON output. |
| `VS_OUTPUT_DIR` | `VS/output` | Load stitched JSON + validate per-page JSON. |
| `VS_LOG_FILE` | `logs/pipeline.log` | Single logfile; no stdout handlers. |

### Retry & Fatal Behavior

`VS_MAX_ATTEMPTS`-bounded T→V retries per page. First fatal page stops the run; partial progress still stitched to disk. Exit code is non-zero on any fatal failure.

---

## Testing

Run the unit suite (no network or API key required; the OpenAI client is faked):

```bash
uv run pytest
```

---

## Cost

### Projection

Gemini Pro 3.1 - $0.127/page

Gemini Flash 3.5 - $0.0271/page

Claude Fable 5 - $0.2845/page

Claude Sonnet 5 - $0.0675/page

Claude Sonnet 4.6 - $0.04755/page

GPT 5.5 - $0.199/page (extracts 1 line json)

GPT 5.4 - $0.0351/page (extracts 1 line json)

GPT 5.4 Mini - $0.009/page (extracts 1 line json, did not follow instructions and conform to schema, not a usable json output)

Kimi K2.6 - $0.062/page (failed at extracting the first page due to rate limits, extracts 1 line json)

Kimi K2.7 Code - $0.04965/page (did not get v. variants, extracts 1 line json, removed POS from trailing text)

**NOTE:** The ideal cost to performance seems to be between $0.01-$0.05, Gemini 3.5 Flash, Claude Sonnet 4.6, GPT 5.4, and Kimi K2.7 Code

### Runtime Tracking

The pipeline records the **actual** OpenRouter cost of every LLM call and writes the tally to `logs/pipeline.log`.

- **Per call** — `extract_json` (`src/transform/client.py`) logs `page N cost=$... (page_running=$... total=$... calls=...)` at INFO level (failed retry attempts are billed and counted).
- **Final total** — a `COST SUMMARY total=$... across N LLM calls (M pages)` line is emitted from a `finally` block in `src.main.main`, so it lands in the logfile on success, fatal page failure, stitch failure, or `sys.exit(1)`.
- **Parsing** — cost is parsed from OpenRouter's `x-or-cost-*` headers, falling back to a `usage.cost` body field. Unparseable calls count as `$0` and log a WARNING.
- **Concurrency** — the accumulator (`src/common/cost.py`) is process-global and lock-protected, so the tally is correct under `--batch-size` parallelism.
- **Opt-out** — set `VS_TRACK_COST=false` to disable tracking entirely.