"""Project-wide validation error type.

Raised by any pipeline stage (validate, model validators, future load
stage) when a payload fails a runtime quality gate. Kept in `src.common`
so both the Pydantic model layer and the validate stage can import it
without a circular dependency.
"""

from __future__ import annotations


class ValidationError(Exception):
    """Raised when a transformed payload fails validation."""