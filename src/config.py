from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


def sanitize_model(model: str) -> str:
    """Make a model id safe for filenames by replacing ``/`` with ``-``."""
    return model.replace("/", "-")


class Settings(BaseSettings):
    """Typed runtime config; secrets from process env (via `op run`), defaults from `VS_*`."""

    model_config = SettingsConfigDict(
        env_file=None,
        env_prefix="VS_",
        extra="ignore",
    )

    openai_api_key: SecretStr = Field(
        default=SecretStr(""), validation_alias="OPENAI_API_KEY"
    )
    openai_base_url: str = Field(
        default="https://openrouter.ai/api/v1", validation_alias="OPENAI_BASE_URL"
    )
    model: str = Field(
        default="anthropic/claude-sonnet-4.6", validation_alias="MODEL"
    )

    volume: int = 1
    data_dir: Path = Path("VS")
    image_dpi: int = 200
    column_layout: Literal["vertical", "horizontal"] = "vertical"
    strip_ocr_prefix: bool = True
    raw_output_dir: Path = Path("test/data/transform/output")
    output_dir: Path = Path("VS/output")
    log_file: Path = Path("logs/pipeline.log")
    layout_tolerance: float = Field(
        15.0,
        description="Fudge px subtracted from `headword_delta` for the layout heuristic; accounts for scan skew/tilt.",
    )
    headword_delta: float = Field(
        36.0,
        description="Calibrated min |Δx| px between the first two text lines on a headword page at 200 DPI.",
    )
    mode: Literal["debug", "running"] = Field(
        "running",
        description="Pipeline execution mode: 'debug' also persists raw and annotated per-page artifacts; both modes persist successful pages immediately.",
    )
    request_timeout_seconds: float = Field(
        120.0,
        gt=0,
        description="OpenAI-compatible API request timeout in seconds.",
    )
    grounding_threshold: float = Field(
        0.9,
        description=(
            "Min needle-coverage token overlap ratio for fuzzy "
            "`trailing_text` grounding. Consulted only when strict "
            "substring fails and trailing_text has ≥ "
            "`grounding_min_tokens` tokens."
        ),
    )
    grounding_min_tokens: int = Field(
        5,
        description=(
            "Min token count for `trailing_text` to be eligible for the "
            "fuzzy fallback. Below this, strict substring only. With "
            "threshold=0.9, 5-9 token text needs 100% coverage."
        ),
    )
    track_cost: bool = Field(
        True,
        description="Track per-call OpenRouter cost via response headers and emit a running tally plus a final COST SUMMARY line.",
    )
    reasoning_effort: Literal[
        "none", "minimal", "low", "medium", "high", "xhigh", "max"
    ] | None = Field(
        None,
        description="Optional OpenRouter reasoning effort sent to supported models.",
    )

    def pdf_path(self) -> Path:
        return self.data_dir / "columns" / f"VS{self.volume}-1col.pdf"

    def ocr_txt_path(self) -> Path:
        return self.data_dir / "sanitized_ocr" / f"VS{self.volume}-1col-googlevision.txt"

    def raw_page_path(self, page: int, model: str) -> Path:
        """Transform-layer raw_json path (debug mode only / first attempt)."""
        return self.raw_output_dir / f"VS{self.volume}_page_{page:03d}_{sanitize_model(model)}.json"

    def validated_pages_dir(self) -> Path:
        """Per-page validated JSON directory (debug mode only)."""
        return self.output_dir / f"vol_{self.volume}" / "pages"

    def validated_page_path(self, page: int, model: str) -> Path:
        """Validate-layer per-page path (debug mode only)."""
        return self.validated_pages_dir() / f"VS{self.volume}_page_{page:03d}_{sanitize_model(model)}.json"

    def dictionary_path(self, model: str) -> Path:
        """Model-specific, incrementally persisted volume path."""
        return self.output_dir / f"vs_{self.volume}_{sanitize_model(model)}.json"

    def failures_path(self) -> Path:
        """Sorted failed-page ledger for the active volume."""
        return self.output_dir / f"vol_{self.volume}" / "failures.txt"


@lru_cache
def get_settings() -> Settings:
    return Settings()
