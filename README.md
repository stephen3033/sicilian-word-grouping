# sicilian-word-grouping

Extract structured dictionary entries from the *Vocabolario Siciliano* PDF
volumes with an OpenAI-compatible vision model.

## How it works

`PDF image + OCR → model extraction → validation → JSON`

For each unfinished printed page, the pipeline:

1. Combines its two physical PDF pages and loads the matching OCR text.
2. Makes one model request for the page.
3. Validates the response and marks entries that need review.
4. Immediately adds the complete page to the final volume JSON.

Pages can run concurrently with `--batch-size`. A failed page does not stop
other pages, and partial pages are never saved.

## Cost and model choice

**Qwen 3.6 is the model of choice.** Set
`MODEL=qwen/qwen3.6-35b-a3b`. Its output is good enough for this project at a
much lower price than Sonnet 4.6, although it runs substantially slower.

### Initial estimates

These are observed estimates, not fixed prices. Cost varies with model output
and provider pricing.

| Model | Estimated cost/page | Observation |
| --- | ---: | --- |
| **Qwen 3.6** | **$0.02252** | **Recommended value; slower than Sonnet 4.6.** |
| Gemini Pro 3.1 | $0.12700 | — |
| Gemini Flash 3.5 | $0.02710 | Good cost/performance range. |
| Claude Fable 5 | $0.28450 | — |
| Claude Sonnet 5 | $0.06750 | — |
| Claude Sonnet 4.6 | $0.04755 | Good cost/performance range. |
| GPT 5.5 | $0.19900 | Returned one-line JSON. |
| GPT 5.4 | $0.03510 | Good cost/performance range; returned one-line JSON. |
| GPT 5.4 Mini | $0.00900 | Did not reliably follow the schema. |
| Kimi K2.6 | $0.06200 | First-page extraction hit rate limits. |
| Kimi K2.7 Code | $0.04965 | Missed variants and altered trailing text. |

The most promising observed range was about **$0.01–$0.05 per page**.
OpenRouter's actual per-call cost and final total are written to the pipeline
log. Set `VS_TRACK_COST=false` to disable tracking.

### Pages 1–5 E2E comparison

| Model run | Pages | Entries | Runtime | Cost | Calls | Retries | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| **Qwen 3.6, July 22** | 5/5 | 119 | 164s | $0.112596 | 8 | 3 | All pages persisted; acceptable value result. |
| Claude Sonnet 4.6, July 19 | 5/5 | 122 | 102s | $0.395841 | 7 | 2 | Complete pass. |

Qwen cost about **72% less** in this comparison but took about **61% longer**
and omitted three entries found by Sonnet. These dated runs used an earlier
retrying runner; the current pipeline makes one request per page.

## Usage

`.env` supplies `OPENAI_API_KEY`; the examples resolve it through 1Password.
Choose exactly one page selector:

| Selector | Example |
| --- | --- |
| Inclusive range | `--start 1 --end 20` |
| Page list | `--pages 3 7 9` |
| One page per file line | `--pages-file pages.txt` |

```bash
op run --env-file=.env -- uv run sicilian-word-grouping \
  --start 1 --end 20 --batch-size 5
```

`--batch-size` sets the maximum concurrent pages and defaults to `1`.
`--mode debug` also saves raw responses and annotated per-page JSON.

Requested pages are deduplicated and sorted. Pages already in the final JSON
are skipped without making a model request.

## Output and failures

| Output | Location |
| --- | --- |
| Final dictionary | `VS/output/vs_<volume>_<model>.json` |
| Failed-page list | `VS/output/vol_<volume>/failures.txt` |
| Debug page JSON | `VS/output/vol_<volume>/pages/` |
| Raw responses | `test/data/transform/output/` |

Successful pages are saved immediately. Failed pages are recorded as sorted,
unique page numbers, and the process exits with status `1` after all requested
work finishes.

Rerun volume 1 failures with:

```bash
op run --env-file=.env -- uv run sicilian-word-grouping \
  --pages-file VS/output/vol_1/failures.txt --batch-size 5
```

A later success removes the page from `failures.txt`. Existing pre-v2 final
JSON is preserved with a `.legacy.json` suffix.

## Entries and review

Each entry contains the model-extracted `headword`, `variants`, and
`trailing_text`, plus its volume, page numbers, review status, and review
reason.

| Status | Meaning |
| --- | --- |
| `passed` | No quality findings. |
| `machine` | One text finding that may be machine-correctable. |
| `human` | Multiple text findings or a page-layout concern. |

Review findings do not change or discard extracted text. Malformed JSON, an
invalid response schema, missing OCR, provider errors, and timeouts fail the
whole page.

## Configuration

Defaults live in `src/config.py`. Common settings are:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | — | API key. |
| `OPENAI_BASE_URL` | `https://openrouter.ai/api/v1` | OpenAI-compatible endpoint. |
| `MODEL` | `anthropic/claude-sonnet-4.6` | Model identifier; override with Qwen 3.6 as recommended above. |
| `VS_VOLUME` | `1` | Active dictionary volume. |
| `VS_DATA_DIR` | `VS` | PDF and OCR source root. |
| `VS_OUTPUT_DIR` | `VS/output` | Final and per-page outputs. |
| `VS_MODE` | `running` | `running` or `debug`. |
| `VS_REQUEST_TIMEOUT_SECONDS` | `120` | Request timeout. |
| `VS_TRACK_COST` | `true` | Track OpenRouter costs. |
| `VS_LOG_FILE` | `logs/pipeline.log` | Pipeline logfile. |

Rendering, OCR, grounding, and layout thresholds can also be adjusted through
the `VS_*` settings in `src/config.py`.

## Logs and tests

Pipeline logs go only to `VS_LOG_FILE`, keeping console output concise. View
the default log with:

```bash
cat logs/pipeline.log
```

Run the test suite with:

```bash
uv run pytest
```
