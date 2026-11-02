"""Inspect promotion framing and record separately attributed claim evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402


def main(argv=None):
    from fipd.analysis.promotion_audit import audit_promotion

    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--text", default="", help="Exact promotion transcript")
    inputs.add_argument("--text-file", type=Path, help="UTF-8 promotion transcript")
    parser.add_argument("--caption", default="", help="Original caption, kept separate")
    parser.add_argument("--as-of", help="Evidence cutoff date, YYYY-MM-DD")
    parser.add_argument("--evidence", type=Path, help="JSON list of reviewer-supplied claim evidence")
    parser.add_argument("--framing-artifact", type=Path,
                        help="Optional local reviewed-data multi-label SVM artifact")
    args = parser.parse_args(argv)
    try:
        if args.text_file and args.text_file.stat().st_size > 400_000:
            raise ValueError("The transcript file must be smaller than 400 KB.")
        text = args.text_file.read_text(encoding="utf-8-sig") if args.text_file else args.text
        evidence = None
        if args.evidence:
            if args.evidence.stat().st_size > 1_000_000:
                raise ValueError("The evidence file must be smaller than 1 MB.")
            evidence = json.loads(args.evidence.read_text(encoding="utf-8-sig"))
        result = audit_promotion(text, args.caption, as_of=args.as_of, evidence=evidence)
        if args.framing_artifact:
            from fipd.training.framing import load_framing_artifact, predict_framing

            model = load_framing_artifact(args.framing_artifact)
            result["learned_framing"] = predict_framing(model, text, args.caption)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, OSError, ImportError) as exc:
        print(f"Promotion audit stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
