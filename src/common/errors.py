"""Project-wide validation error type, shared by model + validate layers."""

from __future__ import annotations


class ValidationError(Exception):
    """Raised when a transformed payload fails a runtime quality gate."""
