"""Unit tests for src.config path helpers + sanitize_model."""

from __future__ import annotations

from pathlib import Path

from src.config import Settings, sanitize_model


class TestSanitizeModel:
    def test_replaces_slashes_with_hyphens(self):
        assert sanitize_model("anthropic/claude-sonnet-4.6") == "anthropic-claude-sonnet-4.6"

    def test_leaves_models_without_slashes_untouched(self):
        assert sanitize_model("claude-sonnet-4.6") == "claude-sonnet-4.6"

    def test_idempotent(self):
        once = sanitize_model("a/b/c")
        assert sanitize_model(once) == once


class TestSettingsPaths:
    def test_raw_page_path_uses_raw_output_dir_with_padded_page_and_sanitized_model(
        self, tmp_path: Path
    ):
        s = Settings(raw_output_dir=tmp_path / "raw", output_dir=tmp_path / "out")
        p = s.raw_page_path(7, "anthropic/claude-sonnet-4.6")
        assert p == tmp_path / "raw" / "VS1_page_007_anthropic-claude-sonnet-4.6.json"

    def test_validated_pages_dir_nests_under_output_dir(self, tmp_path: Path):
        s = Settings(output_dir=tmp_path)
        assert s.validated_pages_dir() == tmp_path / "vol_1" / "pages"

    def test_validated_page_path_format(self, tmp_path: Path):
        s = Settings(output_dir=tmp_path, volume=3)
        p = s.validated_page_path(42, "google/gemini-3.5-flash")
        assert p == tmp_path / "vol_3" / "pages" / "VS3_page_042_google-gemini-3.5-flash.json"

    def test_dictionary_path_format(self, tmp_path: Path):
        s = Settings(output_dir=tmp_path, volume=2)
        p = s.dictionary_path("moonshotai/kimi-k2.7-code")
        assert p == tmp_path / "vs_2_moonshotai-kimi-k2.7-code.json"

    def test_failures_path_is_volume_specific(self, tmp_path: Path):
        s = Settings(output_dir=tmp_path, volume=2)
        assert s.failures_path() == tmp_path / "vol_2" / "failures.txt"

    def test_mode_default_is_running(self):
        s = Settings()
        assert s.mode == "running"

    def test_mode_accepts_debug(self):
        s = Settings(mode="debug")
        assert s.mode == "debug"

    def test_request_timeout_defaults_to_120_seconds(self):
        s = Settings()
        assert s.request_timeout_seconds == 120.0

    def test_request_timeout_is_overridable(self):
        s = Settings(request_timeout_seconds=45)
        assert s.request_timeout_seconds == 45

    def test_retry_configuration_is_removed(self):
        s = Settings()
        assert not hasattr(s, "max_attempts")
        assert not hasattr(s, "raw_retry_page_path")
