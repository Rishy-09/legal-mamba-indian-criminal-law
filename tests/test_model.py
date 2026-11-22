import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model import BNSMambaConfig, BNSMambaModel


class TestMambaModel(unittest.TestCase):

    def test_forward_shape(self):
        config = BNSMambaConfig(
            vocab_size=128,
            d_model=32,
            n_layers=2,
        )

        model = BNSMambaModel(config)
        tokens = torch.randint(0, 128, (2, 16))

        with torch.no_grad():
            logits = model(tokens)

        self.assertEqual(logits.shape, (2, 16, 128))

    def test_model_has_trainable_parameters(self):
        config = BNSMambaConfig(
            vocab_size=128,
            d_model=32,
            n_layers=2,
        )

        model = BNSMambaModel(config)

        trainable = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

        self.assertGreater(trainable, 0)

    def test_forward_output_is_finite(self):
        config = BNSMambaConfig(
            vocab_size=64,
            d_model=32,
            n_layers=2,
        )

        model = BNSMambaModel(config)
        tokens = torch.randint(0, 64, (1, 8))

        with torch.no_grad():
            logits = model(tokens)

        self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()