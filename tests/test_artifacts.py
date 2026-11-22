import csv
import unittest
from pathlib import Path

from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]


class TestRepositoryArtifacts(unittest.TestCase):

    def test_tokenizer_exists_and_loads(self):
        path = ROOT / "data" / "tokenizer.json"

        self.assertTrue(path.exists())

        tokenizer = Tokenizer.from_file(str(path))

        self.assertEqual(tokenizer.get_vocab_size(), 8192)

    def test_training_corpus_exists(self):
        path = ROOT / "data" / "final_train_v3.txt"

        self.assertTrue(path.exists())
        self.assertGreater(path.stat().st_size, 0)

    def test_loss_history_is_valid(self):
        path = ROOT / "results" / "loss_history.csv"

        self.assertTrue(path.exists())

        with path.open(newline="") as file:
            rows = list(csv.DictReader(file))

        self.assertEqual(len(rows), 300)

        steps = [int(row["step"]) for row in rows]
        losses = [float(row["loss"]) for row in rows]

        self.assertEqual(steps, list(range(300)))
        self.assertTrue(all(loss > 0 for loss in losses))

    def test_training_loss_improved(self):
        path = ROOT / "results" / "loss_history.csv"

        with path.open(newline="") as file:
            rows = list(csv.DictReader(file))

        initial_loss = float(rows[0]["loss"])
        final_loss = float(rows[-1]["loss"])

        self.assertLess(final_loss, initial_loss)


if __name__ == "__main__":
    unittest.main()