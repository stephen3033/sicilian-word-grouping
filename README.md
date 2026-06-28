# sicilian-word-grouping

Extract and group headwords from all variants of Sicilian dialect using the [Vocabolario Siciliano](https://it.wikipedia.org/wiki/Vocabolario_Siciliano) PDF volumes and OpenAI-compatible vision-language models (Ollama, OpenRouter, etc.).

## Architecture Overview

The system is split into four isolated, data-driven layers:

1. **Extract (E):** Handles raw file ingestion. It splits document pages into high-resolution visual layouts (Base64 images) and extracts character-accurate token data from the underlying text layer concurrently.
2. **Transform (T):** Maps layout data and token arrays into structured model payloads, leveraging the OpenRouter API for multimodal schema extraction.
3. **Validate (V):** Instantiates a runtime quality gate. It infers reference-free metrics (like schema syntax compliance and OCR semantic grounding scores) to catch hallucinations on the fly without needing pre-existing golden datasets.
4. **Load (L):** Manages structural persistence, logging clean extracted JSON outputs alongside their analytical evaluation metrics.

---

## Directory Structure

The project follows a modern `src/` layout pattern to separate execution modules from project configurations.

```text
.
├── pyproject.toml         # Project metadata and dependencies managed by uv
├── run.py                 # Main CLI application entry point
├── .env.template          # Environment variables to configure
├── .gitignore             # Ignore useless files and the VS (This should NEVER be published)
├── data/
│   ├── input/             # Source multi-page documents
│   └── output/            # Extracted records and runtime quality metrics
└── src/
    ├── __init__.py
    ├── main.py            # Pipeline orchestrator (Sequences E -> T -> V -> L)
    ├── config.py          # Application configuration and environment guard
    │
    ├── extract/           # [Active] Layer 1: File ingestion & text extraction
    │   ├── __init__.py
    │   ├── pdf_processor.py
    │   └── text_engine.py
    │
    ├── transform/         # [Active] Layer 2: Prompt compilation & LLM interface
    │   ├── __init__.py
    │   ├── client.py
    │   └── prompter.py
    │
    ├── validate/          # [Skeleton] Layer 3: Runtime syntax & grounding metrics
    │   └── __init__.py    # (Stubbed for core evaluation logic development)
    │
    └── load/              # [Skeleton] Layer 4: Storage & performance trace logging
        └── __init__.py    # (Stubbed for data persistence hooks)