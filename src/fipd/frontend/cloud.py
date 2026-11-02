"""Explicit cloud-startup provisioning, separate from offline inference.

Only the Community Cloud entrypoint calls this helper. Model downloads run in a
short-lived process so preparation does not retain a second copy of the models.
No classifier is trained or downloaded here.
"""
from __future__ import annotations

import logging
from pathlib import Path
import subprocess
import sys
import threading

log = logging.getLogger(__name__)
_PREPARE_LOCK = threading.Lock()
OCR_FILES = ("craft_mlt_25k.pth", "devanagari.pth")
SPEECH_FILES = ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt",
                "preparation.json")


def _present(directory: Path, names: tuple[str, ...]) -> bool:
    return all((directory / name).is_file() and (directory / name).stat().st_size > 0
               for name in names)


def _run(command: list[str], root: Path) -> None:
    try:
        subprocess.run(command, cwd=root, check=True, timeout=900)
    except (subprocess.SubprocessError, OSError) as exc:
        log.exception("Cloud demo model preparation failed")
        raise RuntimeError("Cloud demo model preparation failed") from exc


def prepare_cloud_assets(root: Path, *, speech: bool = False) -> None:
    """Prepare public OCR weights, then speech only when its page is visited.

EasyOCR checks its upstream weight checksums. Speech preparation records the
resolved upstream revision and SHA-256 hashes using the existing project CLI.
Uploads and captions are never passed to either download process.
"""
    # Sessions opening different pages can miss different cache keys together.
    # Serialize setup to avoid competing downloads and model initialization.
    with _PREPARE_LOCK:
        _prepare_cloud_assets(Path(root).resolve(), speech=speech)


def _prepare_cloud_assets(root: Path, *, speech: bool) -> None:
    ocr = root / "data/models/easyocr"
    if not _present(ocr, OCR_FILES):
        _run([
            sys.executable, "-c",
            "import sys, torch, easyocr; torch.set_num_threads(2); "
            "easyocr.Reader(['en', 'hi'], gpu=False, model_storage_directory=sys.argv[1], "
            "download_enabled=True, verbose=False)",
            str(ocr),
        ], root)
        if not _present(ocr, OCR_FILES):
            raise RuntimeError("OCR preparation did not produce the required files")
    if speech and not _present(root / "data/models/faster-whisper-base", SPEECH_FILES):
        _run([sys.executable, str(root / "scripts/14_analyze_media.py"), "--prepare-model"], root)
        if not _present(root / "data/models/faster-whisper-base", SPEECH_FILES):
            raise RuntimeError("Speech preparation did not produce the required files")
