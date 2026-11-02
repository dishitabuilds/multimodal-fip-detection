#!/usr/bin/env python3
"""Collect a 30-item AMFI pilot as unknown candidates pending human review.

    python scripts/01b_collect_legitimate.py --limit 30

Real source pages and raw filename provenance are retained. Labels, image roles
and task relevance require manual review before curation/training.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402

from fipd.collection.legitimate import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
