"""OCR command checks must stay offline; failed OCR must remain retryable."""

from __future__ import annotations

import builtins
import importlib.util
import io
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, call, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fipd.enrichment.ocr import (  # noqa: E402
    EasyOCREngine, OCRCache, OCREngine, OCRResult, ocr_records,
)
from fipd.schema.records import FactCheckRecord, ImageAsset  # noqa: E402


def load_script():
    spec = importlib.util.spec_from_file_location("ocr_cli", ROOT / "scripts/03_ocr.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CountingEngine(OCREngine):
    name = "offline"

    def __init__(self, failure=False):
        super().__init__()
        self.loads = 0
        self.failure = failure

    def load(self):
        self.loads += 1
        if self.failure:
            raise RuntimeError("local model files missing; downloads disabled")

    def _run(self, path):
        return ["GUARANTEED MONTHLY RETURNS"], [0.9]


def record(path="images/a.jpg"):
    return FactCheckRecord(source="offline", source_id="1", url="https://example.com/1",
                           images=[ImageAsset(url="https://example.com/a.jpg", local_path=path,
                                              role="creative")])


def stats(errors=0):
    return {"images": 1, "ocr_run": 1, "cache_hits": 0, "skipped_annotated": 0,
            "missing": 0, "errors": errors, "garbage": 0}


class OcrCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cli = load_script()

    def patch_cli(self, stack, argv, engine, batch=None):
        stack.enter_context(patch.object(sys, "argv", ["03_ocr.py", *argv]))
        stack.enter_context(patch.object(self.cli.Path, "mkdir"))
        stack.enter_context(patch.object(self.cli, "read_all", return_value=[record()]))
        log = stack.enter_context(patch.object(self.cli, "setup", return_value=Mock()))
        stack.enter_context(patch.object(self.cli, "banner"))
        get_engine = stack.enter_context(patch.object(self.cli, "get_engine", return_value=engine))
        run = stack.enter_context(patch.object(self.cli, "ocr_records", side_effect=batch,
                                              return_value=stats()))
        write_records = stack.enter_context(patch.object(self.cli, "write_records"))
        write_json = stack.enter_context(patch.object(self.cli, "write_json"))
        stack.enter_context(redirect_stdout(io.StringIO()))
        return get_engine, run, write_records, write_json, log.return_value

    def test_list_backends_does_not_construct_load_or_import_engines(self):
        constructors = {name: Mock(side_effect=AssertionError("must not construct engines"))
                        for name in ("easyocr", "paddleocr", "tesseract")}
        output = io.StringIO()
        real_import = builtins.__import__

        def offline_import(name, *args, **kwargs):
            if name.split(".")[0] in {"easyocr", "paddleocr", "pytesseract", "torch"}:
                raise AssertionError(f"backend listing tried to import {name}")
            return real_import(name, *args, **kwargs)

        with patch.object(self.cli, "BACKENDS", constructors), \
                patch.object(self.cli.importlib.util, "find_spec", return_value=object()) as find, \
                patch.object(self.cli.shutil, "which", return_value=None), \
                patch.object(builtins, "__import__", side_effect=offline_import), \
                patch.object(self.cli, "get_engine", side_effect=AssertionError("must not load")), \
                patch.object(sys, "argv", ["03_ocr.py", "--list-backends"]), \
                redirect_stdout(output):
            self.assertEqual(self.cli.main(), 0)
        find.assert_has_calls([call("easyocr"), call("paddleocr"), call("pytesseract")])
        for constructor in constructors.values():
            constructor.assert_not_called()
        self.assertIn("uninitialized", output.getvalue())
        self.assertIn("binary NOT on PATH", output.getvalue())
        self.assertIn("OCR quality were not checked", output.getvalue())

    def test_initialization_failure_is_nonzero_without_success_outputs(self):
        engine = CountingEngine(failure=True)
        with ExitStack() as stack:
            _, run, write_records, write_json, log = self.patch_cli(stack, [], engine)
            self.assertEqual(self.cli.main(), 1)
        self.assertEqual(engine.loads, 1)
        self.assertFalse(engine._loaded)
        run.assert_not_called()
        write_records.assert_not_called()
        write_json.assert_not_called()
        self.assertIn("Could not initialize", log.error.call_args.args[0])

    def test_separate_output_keeps_experiment_cache_and_records_isolated(self):
        with ExitStack() as stack:
            _, run, write_records, write_json, _ = self.patch_cli(
                stack, ["--in-dir", "data/pilot-input", "--out-dir", "data/pilot-ocr"], CountingEngine())
            self.assertEqual(self.cli.main(), 0)
        self.assertEqual(run.call_args.kwargs["cache_path"], Path("data/pilot-ocr/ocr_cache.json"))
        self.assertEqual(write_records.call_args.args[1], Path("data/pilot-ocr/records.jsonl"))
        self.assertEqual(write_json.call_args.args[1], Path("data/pilot-ocr/ocr_report.json"))

    def test_output_cannot_replace_input_records(self):
        with ExitStack() as stack:
            _, run, write_records, _, _ = self.patch_cli(
                stack, ["--in-dir", "data/input", "--out-dir", "data/input"], CountingEngine())
            with self.assertRaises(SystemExit) as result:
                self.cli.main()
        self.assertEqual(result.exception.code, 2)
        run.assert_not_called()
        write_records.assert_not_called()

    def test_options_and_thread_cap_are_applied_before_one_initialization(self):
        engine = CountingEngine()
        torch = ModuleType("torch")
        torch.set_num_threads = Mock()

        def batch(records, active_engine, **kwargs):
            torch.set_num_threads.assert_called_once_with(2)
            self.assertTrue(active_engine._loaded)
            active_engine.read("a.jpg")
            active_engine.read("b.jpg")
            return stats()

        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, torch=torch))
            get_engine, _, _, _, _ = self.patch_cli(
                stack, ["--model-dir", "local-models", "--no-download", "--threads", "2"],
                engine, batch,
            )
            self.assertEqual(self.cli.main(), 0)
        get_engine.assert_called_once_with("easyocr", languages=["en", "hi"], gpu=False,
                                           download_enabled=False,
                                           model_storage_directory="local-models")
        self.assertEqual(engine.loads, 1)

    def test_per_image_errors_mark_report_as_failed_and_exit_nonzero(self):
        with ExitStack() as stack:
            _, _, _, write_json, _ = self.patch_cli(
                stack, [], CountingEngine(), lambda *a, **kw: stats(errors=1),
            )
            self.assertEqual(self.cli.main(), 1)
        report = write_json.call_args.args[0]
        self.assertEqual(report["status"], "partial_failure")
        self.assertEqual(report["errors"], 1)


class OcrEngineTests(unittest.TestCase):
    def test_easyocr_reader_gets_offline_options_and_loads_once(self):
        easyocr = ModuleType("easyocr")
        easyocr.Reader = Mock()
        easyocr.Reader.return_value.readtext.return_value = [[None, "Readable promotion", 0.9]]
        with patch.dict(sys.modules, easyocr=easyocr):
            engine = EasyOCREngine(languages=["en", "hi"], model_storage_directory="models",
                                   download_enabled=False, gpu=False)
            engine.initialize()
            self.assertFalse(engine.read("a.jpg").error)
            self.assertFalse(engine.read("b.jpg").error)
        easyocr.Reader.assert_called_once_with(["en", "hi"], gpu=False,
                                               model_storage_directory="models",
                                               download_enabled=False, verbose=False)

    def test_successful_cache_is_reused_but_old_failures_are_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            (data_dir / "images").mkdir()
            (data_dir / "images/a.jpg").write_bytes(b"offline image placeholder")
            cache_path = data_dir / "cache.json"
            cache = OCRCache(cache_path)
            cache.put("images/a.jpg", OCRResult(backend="offline", error="load failed"))
            cache.flush()

            engine = CountingEngine()
            first_record = record()
            first = ocr_records([first_record], engine, data_dir=data_dir, cache_path=cache_path)
            self.assertEqual(first["ocr_run"], 1)
            self.assertEqual(first["cache_hits"], 0)
            self.assertEqual(first["errors"], 0)
            self.assertIn("GUARANTEED", first_record.images[0].ocr_text)
            self.assertFalse(OCRCache(cache_path).get("images/a.jpg").error)

            second_engine = CountingEngine(failure=True)
            second = ocr_records([record()], second_engine, data_dir=data_dir,
                                 cache_path=cache_path)
            self.assertEqual(second["cache_hits"], 1)
            self.assertEqual(second["ocr_run"], 0)
            self.assertEqual(second_engine.loads, 0)


if __name__ == "__main__":
    unittest.main()
