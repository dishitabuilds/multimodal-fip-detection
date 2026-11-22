from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fipd.models.message_text import duplicate_key, normalize_message


def test_message_normalization_should_hide_exact_url_identity():
    assert normalize_message("Claim https://targetwinner.com/prize") == "claim  URL"
    assert normalize_message("Claim https://other.example.com/prize") == "claim  URL"


def test_duplicate_key_should_group_numeric_template_variations():
    assert duplicate_key("Claim £500 at https://a.com/123") == duplicate_key(
        "Claim £750 at https://b.com/987")


def test_training_reader_should_exclude_spam_and_conflicting_duplicates(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    from train_smishing_text import read_training

    path = tmp_path / "messages.csv"
    path.write_text('LABEL,TEXT\nham,Meeting at noon\nspam,Buy now\n'
                    'Smishing,Claim 500 at https://a.com/1\n'
                    'smishing,Claim 700 at https://b.com/2\n')
    rows = read_training(path)
    assert len(rows) == 2
    assert sorted(label for _, label in rows) == [0, 1]

    path.write_text('LABEL,TEXT\nham,Claim 500 at https://a.com/1\n'
                    'Smishing,Claim 700 at https://b.com/2\n')
    import pytest
    with pytest.raises(ValueError, match="conflicting labels"):
        read_training(path)
