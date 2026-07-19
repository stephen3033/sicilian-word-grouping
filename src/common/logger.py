"""Pipeline logging: single FileHandler, `src` level mode-driven, libraries at ERROR.

Format embeds `%(funcName)s` between logger name and message:
``%(asctime)s [%(levelname)s] %(name)s.%(funcName)s: %(message)s``.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path

from src.common.errors import ValidationError

_QUIET_LIB_LOGGERS = ("openai", "httpx", "pydantic")


def configure_logging(
    log_file: Path, src_level: int = logging.INFO
) -> None:
    """Configure root + `src` loggers with one FileHandler. Idempotent.

    `src_level` is the level for the `src` package logger: `logging.DEBUG`
    in debug mode (verbose per-entry / per-page traces), `logging.INFO` in
    running mode (per-page summaries only). Root logger stays at WARNING
    and the third-party library loggers in `_QUIET_LIB_LOGGERS` are pinned
    to ERROR, so library chatter never reaches the logfile.
    """
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()

    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s.%(funcName)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(handler)
    root.setLevel(logging.WARNING)
    logging.getLogger("src").setLevel(src_level)
    for name in _QUIET_LIB_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)


def log_errors(func):
    """Log any Exception raised by `func` under its module logger, then re-raise.

    Tags the exception with `_vs_logged = True` so outer decorated callers
    skip re-logging. `ValidationError` is logged single-line (no traceback);
    other exceptions get `exc_info=True`.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if not getattr(e, "_vs_logged", False):
                if isinstance(e, ValidationError):
                    logging.getLogger(func.__module__).error("%s failed: %s", func.__name__, e)
                else:
                    logging.getLogger(func.__module__).error(
                        "%s failed", func.__name__, exc_info=True
                    )
                e._vs_logged = True
            raise

    return wrapper
