"""Count config-only architectures on torch's meta device, with no weight downloads.

The dictionaries below contain the published shape fields verified on 2026-10-25:
https://raw.githubusercontent.com/AI4Bharat/Indic-BERT-v1/master/configs/albert_base_config.json
https://huggingface.co/facebook/deit-tiny-patch16-224/blob/main/config.json
https://huggingface.co/WinKawaks/vit-small-patch16-224/blob/main/config.json
https://huggingface.co/distilbert/distilbert-base-multilingual-cased/raw/main/config.json

Meta construction checks the actual installed model implementations without
allocating full weight storage or pretending these are trained checkpoints.
"""

from __future__ import annotations

import importlib.util
import socket
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fipd.models.arms import build_arm  # noqa: E402
from fipd.models.budget import count_parameters, state_dict_size_mb  # noqa: E402
from fipd.models.config import load_model_config  # noqa: E402

INDICBERT = dict(vocab_size=200000, embedding_size=128, hidden_size=768,
                intermediate_size=3072, max_position_embeddings=512, type_vocab_size=2,
                num_attention_heads=12, num_hidden_layers=12, num_hidden_groups=1,
                inner_group_num=1, hidden_act="gelu", hidden_dropout_prob=0,
                attention_probs_dropout_prob=0)
DEIT_TINY = dict(hidden_size=192, intermediate_size=768, num_hidden_layers=12,
                 num_attention_heads=3, image_size=224, patch_size=16, num_channels=3)
VIT_SMALL = dict(hidden_size=384, intermediate_size=1536, num_hidden_layers=12,
                 num_attention_heads=6, image_size=224, patch_size=16, num_channels=3,
                 qkv_bias=True)
DISTIL_MULTILINGUAL = dict(vocab_size=119547, dim=768, hidden_dim=3072,
                          max_position_embeddings=512, n_heads=12, n_layers=6)

HAS_MODELS = all(importlib.util.find_spec(module) for module in ("torch", "transformers"))


@unittest.skipUnless(HAS_MODELS, "optional torch/transformers extras are absent")
class CandidateArchitectureTests(unittest.TestCase):
    def test_exact_current_and_proposed_parameter_counts(self):
        import torch
        import transformers

        candidate = load_model_config(ROOT / "configs/indicbert-deit-tiny.yaml")
        current = load_model_config(ROOT / "configs/model.yaml")
        configurations = {
            "ai4bharat/indic-bert": transformers.AlbertConfig(**INDICBERT),
            "facebook/deit-tiny-patch16-224": transformers.ViTConfig(**DEIT_TINY),
            "distilbert-base-multilingual-cased": transformers.DistilBertConfig(**DISTIL_MULTILINGUAL),
            "WinKawaks/vit-small-patch16-224": transformers.ViTConfig(**VIT_SMALL),
        }

        def local_config(name):
            return configurations[name]

        with patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")), \
                patch.object(transformers.AutoConfig, "from_pretrained", side_effect=local_config), \
                torch.device("meta"):
            expected = {"text_only": 33628674, "image_only": 5635970, "fused": 39858114}
            for arm, parameters in expected.items():
                with self.subTest(arm=arm):
                    model = build_arm(arm, candidate.student, from_scratch=True)
                    self.assertEqual(count_parameters(model), parameters)
                    self.assertTrue(all(parameter.is_meta for parameter in model.parameters()))
                    self.assertLess(parameters, 50_000_000)
                    if arm == "fused":
                        self.assertEqual(count_parameters(model.text_encoder), 33443584)
                        self.assertEqual(count_parameters(model.image_encoder), 5561472)
                        self.assertGreater(state_dict_size_mb(model), 100)
            original = build_arm("fused", current.student, from_scratch=True)
            self.assertEqual(count_parameters(original.text_encoder), 134734080)
            self.assertEqual(count_parameters(original.image_encoder), 21813504)
            self.assertEqual(count_parameters(original), 159654530)
            self.assertGreater(count_parameters(original.text_encoder.backbone.embeddings), 50_000_000)

    def test_research_candidate_keeps_three_seeds_and_original_default(self):
        candidate = load_model_config(ROOT / "configs/indicbert-deit-tiny.yaml")
        current = load_model_config(ROOT / "configs/model.yaml")
        self.assertEqual(candidate.training.seeds, [13, 42, 1337])
        self.assertEqual(candidate.student.hidden_dim, 192)
        self.assertEqual(candidate.student.fusion_heads, 3)
        self.assertEqual(candidate.student.image_encoder, "facebook/deit-tiny-patch16-224")
        self.assertEqual(current.student.text_encoder, "distilbert-base-multilingual-cased")


if __name__ == "__main__":
    unittest.main()
