"""Real CPU tensors exercise the opt-in random smoke without any model downloads."""

from __future__ import annotations

import importlib.util
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fipd.models.arms import build_arm  # noqa: E402
from fipd.models.budget import count_parameters  # noqa: E402
from fipd.models.config import ArchConfig, load_model_config  # noqa: E402
from fipd.models.tiny import ORIGIN, prepare_tiny_assets  # noqa: E402

HAS_MODELS = all(importlib.util.find_spec(module) for module in ("torch", "transformers"))


@unittest.skipUnless(HAS_MODELS, "optional torch/transformers model extras are absent")
class TinySmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import AutoTokenizer

        cls.torch = torch
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(4)
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        # A mistaken request for a published model must fail instead of making
        # this test pass only on machines with connectivity or a populated cache.
        cls.network = patch.object(socket.socket, "connect",
                                   side_effect=AssertionError("offline smoke tried network access"))
        cls.network.start()
        try:
            cls.assets = cls.root / "tiny"
            cls.manifest = prepare_tiny_assets(cls.assets, seed=42)
            cls.cfg = ArchConfig(text_encoder=str(cls.assets / "text"),
                                 image_encoder=str(cls.assets / "image"), hidden_dim=128,
                                 fusion_heads=4, fusion_layers=1)
            cls.tokenizer = AutoTokenizer.from_pretrained(cls.cfg.text_encoder,
                                                         local_files_only=True)
            cls.text = cls.tokenizer(["Check the investment claim", "सेबी पंजीकृत निवेश"],
                                     padding="max_length", truncation=True, max_length=64,
                                     return_tensors="pt")
            cls.pixels = torch.randn(2, 3, 224, 224)
        except BaseException:
            cls.network.stop()
            cls.directory.cleanup()
            torch.set_num_threads(cls.previous_threads)
            raise

    @classmethod
    def tearDownClass(cls):
        cls.network.stop()
        cls.directory.cleanup()
        cls.torch.set_num_threads(cls.previous_threads)

    def inputs(self, arm):
        inputs = {}
        if arm in ("text_only", "fused"):
            inputs.update(input_ids=self.text["input_ids"],
                          attention_mask=self.text["attention_mask"])
        if arm in ("image_only", "fused"):
            inputs["pixel_values"] = self.pixels
        return inputs

    def test_byte_tokenizer_round_trips_english_hindi_without_unknowns(self):
        for text in ("A verified investment plan", "सेबी पंजीकृत निवेश", "SEBI का दावा 40%"):
            with self.subTest(text=text):
                tokens = self.tokenizer.encode(text)
                self.assertNotIn(self.tokenizer.unk_token_id, tokens)
                self.assertEqual(self.tokenizer.decode(tokens, skip_special_tokens=True,
                                                        clean_up_tokenization_spaces=False), text)
        self.assertEqual(len(self.tokenizer), 261)
        self.assertFalse(self.manifest["pretrained"])
        self.assertEqual(self.manifest["weight_origin"], ORIGIN)
        self.assertEqual(self.manifest["text_parameters"], 331392)
        self.assertEqual(self.manifest["image_parameters"], 405504)

    def test_all_arms_have_finite_logits_gradients_and_expected_parameter_counts(self):
        torch = self.torch
        expected = {"text_only": 364674, "image_only": 438786, "fused": 1068034}
        for arm, parameters in expected.items():
            with self.subTest(arm=arm):
                torch.manual_seed(42)
                model = build_arm(arm, self.cfg)
                self.assertEqual(count_parameters(model), parameters)
                self.assertLess(parameters, 50_000_000)
                logits = model(**self.inputs(arm))
                self.assertEqual(tuple(logits.shape), (2, 2))
                self.assertTrue(torch.isfinite(logits).all())
                loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
                loss.backward()
                gradients = [parameter.grad for parameter in model.parameters()
                             if parameter.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
                self.assertTrue(any(gradient.abs().sum() > 0 for gradient in gradients))
                for encoder in (model.text_encoder, model.image_encoder):
                    if encoder is not None:
                        self.assertTrue(any(parameter.grad is not None and
                                            parameter.grad.abs().sum() > 0
                                            for parameter in encoder.parameters()))
                before = model.classifier[-1].weight.detach().clone()
                torch.optim.SGD(model.parameters(), lr=0.01).step()
                self.assertFalse(torch.equal(before, model.classifier[-1].weight))

    def test_trained_checkpoint_reloads_identical_fused_predictions(self):
        torch = self.torch
        model = build_arm("fused", self.cfg)
        inputs = self.inputs("fused")
        loss = torch.nn.functional.cross_entropy(model(**inputs), torch.tensor([0, 1]))
        loss.backward()
        torch.optim.SGD(model.parameters(), lr=0.01).step()
        model.eval()
        with torch.no_grad():
            expected = model(**inputs)
        checkpoint = self.root / "fused.pt"
        torch.save({"state_dict": model.state_dict(), "weight_origin": ORIGIN}, checkpoint)
        restored = build_arm("fused", self.cfg, from_scratch=True)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        restored.load_state_dict(saved["state_dict"])
        restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(restored(**inputs), expected, rtol=0, atol=0)
        self.assertEqual(saved["weight_origin"], ORIGIN)

    def test_asset_generation_refuses_overwrite(self):
        with self.assertRaises(FileExistsError):
            prepare_tiny_assets(self.assets, seed=43)

    def test_smoke_settings_do_not_change_research_defaults(self):
        smoke = load_model_config(ROOT / "configs/tiny-smoke.yaml")
        research = load_model_config(ROOT / "configs/model.yaml")
        self.assertEqual(smoke.training.seeds, [42])
        self.assertEqual(smoke.training.epochs, 1)
        self.assertEqual(smoke.training.batch_size, 4)
        self.assertEqual(smoke.measurement.threads, 4)
        self.assertEqual(smoke.student.text_encoder, "data/models/tiny-smoke/text")
        self.assertGreaterEqual(len(research.training.seeds), 3)
        self.assertEqual(research.student.text_encoder, "distilbert-base-multilingual-cased")


if __name__ == "__main__":
    unittest.main()
