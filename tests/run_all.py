#!/usr/bin/env python3
"""Run every test suite. No network required.

    python tests/run_all.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SUITES = [
    ("collection / filtering / labelling", "tests/unit/test_collection.py"),
    ("dedup / splits / OCR / translit", "tests/unit/test_curation.py"),
    ("budget / metrics / examples", "tests/unit/test_models.py"),
    ("pipeline integration", "tests/integration/test_stages.py"),
]


def main() -> int:
    results = []
    for name, path in SUITES:
        p = ROOT / path
        if not p.exists():
            print(f"SKIP  {name} ({path} not found)")
            continue
        print("\n" + "=" * 66)
        print(f"  {name}")
        print("=" * 66)
        r = subprocess.run([sys.executable, str(p)], cwd=ROOT)
        results.append((name, r.returncode))

    print("\n" + "=" * 66)
    print("  OVERALL")
    print("=" * 66)
    failed = 0
    for name, code in results:
        status = "PASS" if code == 0 else "FAIL"
        failed += code != 0
        print(f"  {status}  {name}")
    print()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
