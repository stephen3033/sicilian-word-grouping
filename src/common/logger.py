"""Pipeline logging configuration.

All pipeline logging uses Python's `logging` module writing to a single
configurable logfile (default `logs/pipeline.log`). No stdout/stderr
handlers are attached so console output stays clean; inspect the logfile
with `cat logs/pipeline.log`.

`configure_logging` is called once at pipeline startup. The root logger
captures WARNING+ from third-party libraries (openai, httpx, pydantic),
while the `src` package logger captures DEBUG+ so granular per-function
step traces land in the logfile without flooding it with library chatter.

`log_errors` is a decorator applied to every pipeline function. It logs
any raised `Exception` (with full traceback) under the function's own
module logger, tags the exception so outer decorated callers skip
re-logging the same error, then re-raises - so you never write a
`raise` statement just for logging.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path


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
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(handler)
    root.setLevel(logging.WARNING)

    # Verbose DEBUG traces for our own code, quiet WARNING+ for libraries.
    logging.getLogger("src").setLevel(logging.DEBUG)


def log_errors(func):
    """Decorator that logs any Exception raised by `func`, then re-raises.

    The innermost decorated function that raises logs ERROR with a full
    traceback under its module logger and tags the exception with
    `_vs_logged = True`. Outer decorated callers see the flag and skip
    re-logging the same exception, avoiding duplicate error entries while
    still propagating the original exception unchanged.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if not getattr(e, "_vs_logged", False):
                logging.getLogger(func.__module__).error(
                    "%s failed", func.__name__, exc_info=True
                )
                e._vs_logged = True
            raise

    return wrapper
