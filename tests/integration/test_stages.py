"""End-to-end run of the stage scripts against the mock server.

Exercises the real CLIs (01 -> 04 -> 05) exactly as a user would invoke them,
so argument parsing, file layout and inter-stage contracts are covered, not
just the library functions underneath.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tests.fixtures.mock_server import ROOT as MOCK, serve  # noqa: E402

PASS = FAIL = 0


def check(cond, msg, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {msg}")
    else:
        FAIL += 1
        print(f"  FAIL  {msg}   {detail}")


def run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *args],
        capture_output=True, text=True, cwd=ROOT,
    )


def main() -> int:
    serve()
    tmp = Path(tempfile.mkdtemp(prefix="fipd-stages-"))
    print(f"\nworkspace: {tmp}\n")

    cfg = tmp / "sources.yaml"
    cfg.write_text(textwrap.dedent(f"""
        defaults:
          min_delay: 0
          max_delay: 0
          out_dir: {tmp}
        sources:
          mockwp:
            kind: wordpress
            enabled: true
            base_url: {MOCK}
            search: [investment, scam, communal]
            per_page: 50
            max_pages: 5
          mockboom:
            kind: article
            enabled: true
            base_url: {MOCK}
            sitemaps: [{MOCK}/sitemap-daily.xml]
            path_filters: ["/fact-check/business", "/scamcheck"]
          disabled:
            kind: wordpress
            enabled: false
            base_url: {MOCK}
    """), encoding="utf-8")

    # ---------------------------------------------------------------
    print("[1] Stage 01 — collect")
    r = run("01_collect.py", "--config", str(cfg), "--out-dir", str(tmp))
    check(r.returncode == 0, "01_collect exits 0", r.stderr[-400:])
    check((tmp / "raw" / "mockwp.jsonl").exists(), "wordpress source wrote JSONL")
    check((tmp / "raw" / "mockboom.jsonl").exists(), "article source wrote JSONL")
    check("Skipping disabled source: disabled" in r.stdout
          and not (tmp / "raw" / "disabled.jsonl").exists(),
          "disabled source skipped and wrote nothing")
    imgs = list((tmp / "images").rglob("*.jpg"))
    check(len(imgs) > 0, f"images downloaded ({len(imgs)})")

    r2 = run("01_collect.py", "--config", str(cfg), "--out-dir", str(tmp))
    n_after = sum(1 for _ in open(tmp / "raw" / "mockwp.jsonl", encoding="utf-8"))
    check(r2.returncode == 0 and n_after == 3, "re-run is idempotent", n_after)

    # ---------------------------------------------------------------
    print("\n[2] Stage 04 — curate")
    r = run("04_curate.py", "--data-dir", str(tmp))
    check(r.returncode == 0, "04_curate exits 0", r.stderr[-400:])
    for name in ("curated.jsonl", "finance.jsonl", "canonical.jsonl", "curation_report.json"):
        check((tmp / "interim" / name).exists(), f"wrote {name}")

    report = json.loads((tmp / "interim" / "curation_report.json").read_text())
    check(report["total_raw"] == 5, "all raw records processed", report["total_raw"])
    check(report["finance_labelled"] == 4, "4 finance records kept",
          report["finance_labelled"])
    check(report["dedup"] is not None, "dedup ran")
    check("ONE CLASS ONLY" in r.stdout, "single-class warning shown to the user")
    check("Majority-class baseline" in r.stdout, "majority baseline reported")

    r = run("04_curate.py", "--data-dir", str(tmp), "--no-dedup")
    check(r.returncode == 0, "--no-dedup works")
    check(json.loads((tmp / "interim" / "curation_report.json").read_text())["dedup"] is None,
          "--no-dedup skips clustering")

    r = run("04_curate.py", "--data-dir", str(tmp))  # restore clusters

    # ---------------------------------------------------------------
    print("\n[3] Stage 05 — split")
    r = run("05_split.py", "--data-dir", str(tmp), "--write-splits")
    check(r.returncode == 0, "05_split exits 0", r.stderr[-400:] + r.stdout[-400:])
    man = tmp / "processed" / "split_manifest.json"
    check(man.exists(), "manifest written")

    data = json.loads(man.read_text())
    check(data["leakage_ok"] is True, "manifest records a passing leakage check")
    check(data["seed"] == 42, "default seed recorded")
    check(len(data["assignment_hash"]) == 16, "assignment hash present")
    check(sum(data["counts"].values()) == data["n_records"], "all records assigned",
          data["counts"])
    check("PASS" in r.stdout, "leakage check reported as PASS")

    # cluster integrity across the frozen split files
    splits = data["splits"]
    clusters = data["clusters"]
    spread = {}
    for uid, s in splits.items():
        spread.setdefault(clusters.get(uid, uid), set()).add(s)
    check(all(len(v) == 1 for v in spread.values()),
          "no cluster spans splits in the written manifest",
          [k for k, v in spread.items() if len(v) > 1])

    r = run("05_split.py", "--data-dir", str(tmp), "--seed", "42")
    data2 = json.loads(man.read_text())
    check(data2["assignment_hash"] == data["assignment_hash"],
          "same seed reproduces the split exactly")

    r = run("05_split.py", "--data-dir", str(tmp), "--ratios", "0.5", "0.3", "0.2")
    check(r.returncode == 0, "custom ratios accepted")
    r = run("05_split.py", "--data-dir", str(tmp), "--ratios", "0.5", "0.3", "0.9")
    check(r.returncode != 0, "ratios that do not sum to 1 are rejected")

    # ---------------------------------------------------------------
    print("\n[4] Stage 03 — OCR CLI surface")
    r = run("03_ocr.py", "--list-backends")
    check(r.returncode == 0, "--list-backends exits 0")
    check(all(b in r.stdout for b in ("paddleocr", "easyocr", "tesseract")),
          "all backends listed")

    r = run("03_ocr.py", "--data-dir", str(tmp), "--backend", "tesseract", "--limit", "1")
    check(r.returncode in (0, 1), "missing OCR engine fails cleanly, no traceback",
          r.stderr[-200:])
    check("Traceback" not in r.stderr, "no unhandled exception when engine absent")

    shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 58)
    print(f"  {PASS} passed, {FAIL} failed")
    print("=" * 58 + "\n")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
