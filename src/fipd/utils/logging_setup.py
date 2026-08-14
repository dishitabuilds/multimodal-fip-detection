"""One logging configuration for every stage script."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def use_utf8_stdout() -> None:
    """Make stdout/stderr able to carry Devanagari.

    A Windows console defaults to cp1252, so the first Hindi headline logged
    during collection raises UnicodeEncodeError and kills the run. Half this
    project's corpus is Hindi, so this is not an edge case.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass  # already wrapped, or not a real stream (pytest capture)


def setup(verbose: bool = False, log_file: str | Path | None = None) -> logging.Logger:
    use_utf8_stdout()
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s %(levelname)-7s %(name)-28s | %(message)s"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S",
                        handlers=handlers, force=True)

    # These libraries are extremely chatty at DEBUG and drown our own logs.
    for noisy in ("urllib3", "PIL", "requests", "charset_normalizer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logging.getLogger("fipd")


def banner(title: str, width: int = 70) -> None:
    log = logging.getLogger("fipd")
    log.info("=" * width)
    log.info(title)
    log.info("=" * width)
