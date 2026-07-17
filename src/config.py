from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed runtime config; secrets come from the process env (via `op run`),
    non-secret defaults from `VS_*` env vars."""

    model_config = SettingsConfigDict(
        env_file=None,
        env_prefix="VS_",
        extra="ignore",
    )

    # --- secrets / connection config (injected by `op run` from .env) ---
    openai_api_key: SecretStr = Field(
        default=SecretStr(""), validation_alias="OPENAI_API_KEY"
    )
    openai_base_url: str = Field(
        default="https://openrouter.ai/api/v1", validation_alias="OPENAI_BASE_URL"
    )
    model: str = Field(
        default="anthropic/claude-sonnet-4.6", validation_alias="MODEL"
    )

    # --- non-secret runtime config (defaults in code; override via VS_* env) ---
    volume: int = 1
    data_dir: Path = Path("VS")
    image_dpi: int = 200
    column_layout: Literal["vertical", "horizontal"] = "vertical"
    strip_ocr_prefix: bool = True
    output_dir: Path = Path("test/data/transform/output")
    log_file: Path = Path("logs/pipeline.log")

    def pdf_path(self) -> Path:
        return self.data_dir / "columns" / f"VS{self.volume}-1col.pdf"

    def ocr_txt_path(self) -> Path:
        return self.data_dir / "OCR_cols" / f"VS{self.volume}-1col-googlevision.txt"


@lru_cache
def get_settings() -> Settings:
    return Settings()