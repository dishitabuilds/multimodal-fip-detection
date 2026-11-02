"""Budget measurements must describe the execution and memory actually measured.

These tests use tiny stand-ins for torch and platform APIs: no model downloads,
GPU, network, or optional model dependencies are needed.
"""

from __future__ import annotations

import importlib.util
import io
import sys
import unittest
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fipd.models import budget  # noqa: E402


def load_script():
    spec = importlib.util.spec_from_file_location("budget_cli", ROOT / "scripts/06_budget.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PeakMemoryTests(unittest.TestCase):
    def test_windows_uses_peak_working_set_not_current_rss(self):
        psutil = ModuleType("psutil")
        psutil.Process = Mock(return_value=SimpleNamespace(
            memory_info=lambda: SimpleNamespace(rss=4 * 1024**2, peak_wset=25 * 1024**2)
        ))
        with patch.object(sys, "platform", "win32"), patch.dict(sys.modules, psutil=psutil):
            self.assertEqual(budget.peak_ram_mb(), 25.0)

    def test_current_rss_is_not_a_peak_fallback(self):
        psutil = ModuleType("psutil")
        psutil.Process = Mock(return_value=SimpleNamespace(
            memory_info=lambda: SimpleNamespace(rss=4 * 1024**2)
        ))
        with patch.object(sys, "platform", "win32"), patch.dict(sys.modules, psutil=psutil):
            self.assertIsNone(budget.peak_ram_mb())

    def test_unix_peak_units(self):
        for platform, native_peak in (("linux", 25 * 1024), ("darwin", 25 * 1024**2)):
            with self.subTest(platform=platform):
                resource = ModuleType("resource")
                resource.RUSAGE_SELF = 0
                resource.getrusage = Mock(return_value=SimpleNamespace(ru_maxrss=native_peak))
                with patch.object(sys, "platform", platform), \
                        patch.dict(sys.modules, resource=resource):
                    self.assertEqual(budget.peak_ram_mb(), 25.0)
                resource.getrusage.assert_called_once_with(resource.RUSAGE_SELF)

    def test_unavailable_peak_leaves_ram_unmeasured(self):
        psutil = ModuleType("psutil")
        psutil.Process = Mock(side_effect=OSError("process counters unavailable"))
        with patch.object(sys, "platform", "win32"), patch.dict(sys.modules, psutil=psutil), \
                self.assertLogs(budget.log, level="WARNING"):
            peak = budget.peak_ram_mb()
        self.assertIsNone(peak)
        checks = budget.check_budget(budget.Measurement(peak_ram_mb=peak), budget.Budget())
        ram = next(check for check in checks if check.name == "peak RAM")
        self.assertEqual(ram.status, "not measured")
        self.assertFalse(ram.ok)

    def test_unsupported_platform_does_not_guess_units(self):
        with patch.object(sys, "platform", "unknown-platform"):
            self.assertIsNone(budget.peak_ram_mb())


class BudgetCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cli = load_script()

    def test_onnx_rejected_before_model_build_or_torch_load(self):
        torch = ModuleType("torch")
        torch.load = Mock(side_effect=AssertionError("must not load ONNX with torch"))
        arms = ModuleType("fipd.models.arms")
        arms.build_arm = Mock(side_effect=AssertionError("must not construct a model"))
        error = io.StringIO()
        with patch.object(sys, "argv", ["06_budget.py", "--checkpoint", "model.ONNX"]), \
                patch.dict(sys.modules, {"torch": torch, "fipd.models.arms": arms}), \
                redirect_stderr(error), self.assertRaises(SystemExit) as raised:
            self.cli.main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("ONNX checkpoints are not supported", error.getvalue())
        torch.load.assert_not_called()
        arms.build_arm.assert_not_called()

    def test_fp32_execution_is_distinct_from_int8_target(self):
        # Simulate a non-fp32 default: the benchmark must cast the module and
        # create fp32 pixels, then report that execution rather than the target.
        param = Mock(requires_grad=True)
        param.numel.return_value = 16
        param.element_size.return_value = 2
        model = Mock()
        model.text_encoder.backbone.config.vocab_size = 37
        model.parameters.return_value = [param]
        model.buffers.return_value = []
        model.float.side_effect = lambda: setattr(param.element_size, "return_value", 4)
        arms = ModuleType("fipd.models.arms")
        arms.build_arm = Mock(return_value=model)
        torch = ModuleType("torch")
        torch.__version__ = "offline-test"
        torch.float32 = "torch.float32"
        torch.set_num_threads = Mock()
        torch.randint = Mock(return_value=object())
        torch.ones_like = Mock(return_value=object())
        torch.randn = Mock(return_value=object())
        torch.no_grad = lambda: nullcontext()
        cfg = SimpleNamespace(
            measurement=SimpleNamespace(batch_size=1, threads=1, image_size=8,
                                        max_text_tokens=4, precision="int8", warmup_runs=1,
                                        timed_runs=3, reference_machine="offline test"),
            student=SimpleNamespace(text_encoder="offline", image_encoder="offline"),
        )
        with patch.object(sys, "argv", ["06_budget.py", "--from-scratch"]), \
                patch.dict(sys.modules, {"torch": torch, "fipd.models.arms": arms}), \
                patch.object(self.cli, "load_model_config", return_value=cfg), \
                patch.object(self.cli.Budget, "from_config", return_value=budget.Budget()), \
                patch.object(self.cli, "peak_ram_mb", return_value=25.0), \
                patch.object(self.cli, "save_report", return_value=Path("unused.json")) as save, \
                patch.object(self.cli, "setup", return_value=Mock()), \
                patch.object(self.cli, "banner"), redirect_stdout(io.StringIO()):
            self.assertEqual(self.cli.main(), 0)
        measurement = save.call_args.args[0]
        self.assertEqual(measurement.conditions["precision"], "fp32")
        self.assertEqual(measurement.conditions["target_precision"], "int8")
        self.assertEqual(measurement.conditions["backend"], "pytorch")
        self.assertEqual(measurement.conditions["weights"], "random")
        self.assertIn("entire process lifetime", measurement.conditions["ram_scope"])
        self.assertEqual(measurement.peak_ram_mb, 25.0)
        self.assertEqual(measurement.disk_mb, 16 * 4 / 1024**2)
        model.float.assert_called_once_with()
        torch.randn.assert_called_once_with(1, 3, 8, 8, dtype=torch.float32)
        torch.randint.assert_called_once_with(0, 37, (1, 4))
        self.assertEqual(model.call_count, 4)  # warm-up plus timed inference

    def test_targets_only_does_not_claim_an_execution_precision(self):
        output = io.StringIO()
        with patch.object(sys, "argv", ["06_budget.py", "--targets-only"]), \
                patch.object(self.cli, "setup", return_value=Mock()), \
                patch.object(self.cli, "banner"), redirect_stdout(output):
            self.assertEqual(self.cli.main(), 0)
        self.assertIn("precision: not measured", output.getvalue())
        self.assertIn("target_precision: int8", output.getvalue())


if __name__ == "__main__":
    unittest.main()
