import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

if any(importlib.util.find_spec(name) is None for name in ("numpy", "torch", "tokenizers")):
    raise unittest.SkipTest("Install the dependencies")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import count_table as module
import prepare

BOS = 0  # prepare.py's tokenizer gives its only special token the first number
SHARED = [5, 6, 7, 8, 9]


def corpus():
    """Ten documents in which five tokens are followed by 10, and five in which they are followed by 11."""
    documents = [[BOS, *SHARED, 10, 12] for _ in range(10)] + [[BOS, *SHARED, 11, 13] for _ in range(5)]
    return np.array([token for document in documents for token in document], dtype=np.uint16)


class TableTest(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory)
        tokenizer = prepare.train_tokenizer(iter(["a few words to train on"] * 20), 300)
        self.tokenizer = self.directory / "tokenizer.json"
        tokenizer.save(str(self.tokenizer))

    def built(self, tokens, floor=4, parts=3):
        text = self.directory / "text"
        text.mkdir(exist_ok=True)
        tokens.tofile(text / "train.bin")
        output = self.directory / f"table-{floor}-{parts}"
        module.build(text, output, self.tokenizer, floor=floor, parts=parts, chunk=7)
        return output

    def candidates_after(self, table, row, position):
        tokens, weights = module.CountTable(table).candidates(torch.tensor([row]))
        return tokens[0, position].view(module.ORDERS, module.K), weights[0, position].view(module.ORDERS, module.K)

    def test_computes_the_same_keys_in_numpy_and_pytorch(self):
        values = np.random.default_rng(0).integers(1, module.VOCABULARY + 1, size=(3, 40)).astype(np.int64)
        for length in module.LENGTHS:
            in_numpy = module.history_keys(values, length)
            in_pytorch = module.history_keys(torch.from_numpy(values), length).numpy()
            self.assertTrue(np.array_equal(in_numpy, in_pytorch))

    def test_gives_different_histories_different_keys(self):
        values = np.arange(1, 2001, dtype=np.int64)
        for length in module.LENGTHS:
            keys = module.history_keys(values, length)
            self.assertEqual(len(np.unique(keys)), len(keys))

    def test_scores_the_tokens_that_follow_a_history_by_how_often_they_do(self):
        table = self.built(corpus())
        row = [BOS, *SHARED]
        tokens, weights = self.candidates_after(table, row, len(row) - 1)
        longest = dict(zip(tokens[5].tolist(), weights[5].tolist()))
        self.assertAlmostEqual(longest[10] * module.ORDERS, 10 / 15, delta=0.02)
        self.assertAlmostEqual(longest[11] * module.ORDERS, 5 / 15, delta=0.02)

    def test_puts_no_weight_on_slots_without_a_candidate(self):
        table = self.built(corpus())
        row = [BOS, *SHARED]
        _, weights = self.candidates_after(table, row, len(row) - 1)
        self.assertEqual((weights[5] > 0).sum().item(), 2)

    def test_lets_a_history_begin_with_bos(self):
        table = self.built(corpus())
        _, weights = self.candidates_after(table, [BOS, 5, 6], 2)
        self.assertGreater(weights[3].sum().item(), 0)

    def test_finds_no_history_that_runs_across_the_start_of_a_document(self):
        table = self.built(corpus())
        # 12 BOS 5: the history of three tokens holds a BOS after its first token.
        _, weights = self.candidates_after(table, [12, BOS, 5], 2)
        self.assertEqual(weights[3].sum().item(), 0)
        self.assertGreater(weights[2].sum().item(), 0)

    def test_finds_no_history_longer_than_the_row_before_it(self):
        table = self.built(corpus())
        _, weights = self.candidates_after(table, [7, 8, 9], 1)
        self.assertEqual(weights[3:].sum().item(), 0)

    def test_keeps_only_histories_seen_often_enough(self):
        table = self.built(corpus(), floor=16)
        row = [BOS, *SHARED]
        _, weights = self.candidates_after(table, row, len(row) - 1)
        self.assertEqual(weights[5].sum().item(), 0)
        self.assertGreater(self.candidates_after(self.built(corpus(), floor=15), row, 1)[1][1].sum().item(), 0)

    def test_gives_every_position_the_tokens_seen_most_often_overall(self):
        table = self.built(corpus())
        tokens, weights = self.candidates_after(table, [12, 13], 0)
        self.assertIn(BOS, tokens[0].tolist())
        self.assertGreater(weights[0].sum().item(), 0)

    def test_counts_the_same_table_whatever_the_parts_and_chunks(self):
        first, second = self.built(corpus(), parts=1), self.built(corpus(), parts=4)
        for name in ("keys5", "tokens5", "scores5", "keys2", "tokens2", "scores2"):
            self.assertTrue(np.array_equal(np.load(first / f"{name}.npy"), np.load(second / f"{name}.npy")))

    def test_averages_the_weights_over_the_history_lengths(self):
        table = self.built(corpus())
        row = [BOS, *SHARED]
        _, weights = self.candidates_after(table, row, len(row) - 1)
        self.assertAlmostEqual(weights[1:].sum().item(), 5 / module.ORDERS, delta=0.1)

    def test_records_what_it_counted(self):
        manifest = json.loads((self.built(corpus()) / "manifest.json").read_text())
        self.assertEqual(manifest["tokens"], len(corpus()))
        self.assertEqual(manifest["floor"], 4)

    def test_refuses_to_write_over_a_directory(self):
        output = self.built(corpus())
        with self.assertRaises(ValueError):
            module.build(self.directory / "text", output, self.tokenizer)


if __name__ == "__main__":
    unittest.main()
