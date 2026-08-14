"""JSONL read/write helpers shared by every stage script."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Iterator

from ..schema.records import FactCheckRecord

log = logging.getLogger(__name__)


def read_records(path: str | Path, strict: bool = False) -> Iterator[FactCheckRecord]:
    """Yield records from one JSONL file, logging and skipping bad lines."""
    path = Path(path)
    if not path.exists():
        log.warning("no such file: %s", path)
        return
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield FactCheckRecord.from_dict(json.loads(line))
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                if strict:
                    raise
                log.warning("%s:%d unparseable (%s)", path.name, i, e)


def read_all(directory: str | Path, pattern: str = "*.jsonl") -> list[FactCheckRecord]:
    """Read every matching JSONL in a directory into one list."""
    directory = Path(directory)
    out: list[FactCheckRecord] = []
    for p in sorted(directory.glob(pattern)):
        n = len(out)
        out.extend(read_records(p))
        log.info("read %d records from %s", len(out) - n, p.name)
    return out


def write_records(records: Iterable[FactCheckRecord], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(rec.to_json() + "\n")
            n += 1
    log.info("wrote %d records -> %s", n, path)
    return n


def append_record(record: FactCheckRecord, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(record.to_json() + "\n")


def existing_uids(path: str | Path) -> set[str]:
    """UIDs already present in a JSONL file — the basis of every resume."""
    path = Path(path)
    if not path.exists():
        return set()
    out: set[str] = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                out.add(json.loads(line)["uid"])
            except (json.JSONDecodeError, KeyError):
                continue
    return out


def write_json(obj, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True),
                    encoding="utf-8")
    return path
