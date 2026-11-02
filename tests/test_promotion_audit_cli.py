"""Exercise the audit's terminal entry point, including fail-closed evidence."""
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("audit_cli", ROOT / "scripts/13_audit_promotion.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def test_cli_explains_promise_without_assigning_factual_verdict(capsys):
    assert cli.main(["--text", "Revenue grew 40%. Guaranteed double returns. Buy now!"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert "return_promise" in {row["label"] for row in result["findings"]}
    assert result["claims"] == []
    assert "label" not in result and "risk_score" not in result


def test_cli_bad_evidence_does_not_emit_partial_result(tmp_path, capsys):
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"not": "a list"}', encoding="utf-8")
    assert cli.main(["--text", "Revenue grew 40%", "--evidence", str(evidence)]) == 1
    output = capsys.readouterr()
    assert not output.out
    assert "Promotion audit stopped:" in output.err


def test_cli_transcript_file_and_caption_remain_separate(tmp_path, capsys):
    transcript = tmp_path / "promotion.txt"
    transcript.write_text("Revenue grew 40%.", encoding="utf-8-sig")
    assert cli.main(["--text-file", str(transcript), "--caption", "Guaranteed double returns"]) == 0
    result = json.loads(capsys.readouterr().out)
    promise = next(row for row in result["findings"] if row["label"] == "return_promise")
    assert promise["field"] == "caption"
