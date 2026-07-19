"""Pipeline logging configuration.

All pipeline logging uses Python's `logging` module writing to a single
configurable logfile (default `logs/pipeline.log`). No stdout/stderr
handlers are attached so console output stays clean; inspect the logfile
with `cat logs/pipeline.log`.

`configure_logging` is called once at pipeline startup. The root logger
captures WARNING+ from third-party libraries, while the `src` package
logger captures DEBUG+ so granular per-function step traces land in the
logfile without flooding it with library chatter. The `openai`, `httpx`,
and `pydantic` loggers are pinned to ERROR so their WARNING-level chatter
(network retries, schema deprecation notes) never reaches the logfile.

The log format embeds `%(funcName)s`, so per-function messages never need
to repeat the function name in the message text:

    2026-07-19 09:12:03 [DEBUG] src.extract.pdf_extractor.extract_page_image: ...

`log_errors` is a decorator applied to every pipeline function. It logs
any raised `Exception` under the function's own module logger, tags the
exception so outer decorated callers skip re-logging the same error, then
re-raises - so you never write a `raise` statement just for logging.

Traceback policy:
- `ValidationError` (a project-level, message-descriptive failure) is
  logged as a single-line ERROR without `exc_info` so the logfile stays
  scannable.
- Any other `Exception` is logged with `exc_info=True` so unexpected
  bugs surface a full traceback for debugging.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path

from src.common.errors import ValidationError

# Third-party libraries whose WARNING+ chatter (network retries, schema
# deprecation notes, etc.) adds noise without signal. Pinned to ERROR.
_QUIET_LIB_LOGGERS = ("openai", "httpx", "pydantic")


def configure_logging(log_file: Path) -> None:
    """Configure the root + `src` loggers with a single FileHandler.

    Idempotent: re-calling clears previously-attached handlers so the
    logfile is never written twice. Parent directories are created.
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

    # Verbose DEBUG traces for our own code, quiet ERROR+ for libraries.
    logging.getLogger("src").setLevel(logging.DEBUG)
    for name in _QUIET_LIB_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)


def log_errors(func):
    """Decorator that logs any Exception raised by `func`, then re-raises.

    The innermost decorated function that raises logs ERROR under its
    module logger and tags the exception with `_vs_logged = True`. Outer
    decorated callers see the flag and skip re-logging the same exception,
    avoiding duplicate error entries while still propagating the original
    exception unchanged.

    `ValidationError` is logged as a single-line ERROR (its message is
    already descriptive); any other `Exception` is logged with
    `exc_info=True` so unexpected bugs surface a full traceback.
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
