import importlib.util
import json
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

if any(importlib.util.find_spec(name) is None for name in ("numpy", "torch")):
    raise unittest.SkipTest("Install the dependencies")

import numpy as np
import torch

REPOSITORY = Path(__file__).parent.parent
VOCABULARY = 16
UPDATES = 8


class TinyRun:
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        data = self.directory / "data"
        data.mkdir()
        np.tile(np.array([0, 1, 2, 3, 4, 5], dtype=np.uint16), 200).tofile(data / "train.bin")
        np.tile(np.arange(VOCABULARY, dtype=np.uint16), 20).tofile(data / "val.bin")
        (data / "tokenizer.json").write_text(json.dumps({"added_tokens": [{"id": 0, "content": "<|bos|>"}]}))
        with (data / "meta.pkl").open("wb") as meta:
            pickle.dump({"vocab_size": VOCABULARY}, meta)
        self.out = self.directory / "out"

    def train(self, *options):
        return subprocess.run(
            [sys.executable, "-u", "train.py", "configs/bundle2.py", "--device=cpu", "--dtype=float32", "--compile=False",
             "--n_layer=1", "--n_embd=32", "--n_head=1", "--block_size=8", "--batch_size=2", "--gradient_accumulation_steps=1",
             f"--max_iters={UPDATES}", f"--lr_decay_iters={UPDATES}", f"--eval_interval={UPDATES}", "--eval_iters=1",
             f"--dataset={self.directory / 'data'}", f"--out_dir={self.out}", *options],
            cwd=REPOSITORY, capture_output=True, text=True, timeout=300,
        )

    def saved(self, *parts):
        return torch.load(self.out.joinpath(*parts, "ckpt.pt"), map_location="cpu")


class CurrentBestTest(TinyRun, unittest.TestCase):
    def train_best(self, *options):
        """train.py's defaults, the current best, cut down to run here."""
        return subprocess.run(
            [sys.executable, "-u", "train.py", "--device=cpu", "--dtype=float32", "--compile=False", "--n_layer=3",
             "--n_embd=32", "--n_head=1", "--block_size=16", "--batch_size=2", "--gradient_accumulation_steps=1",
             "--count_table=", "--eval_iters=1", f"--dataset={self.directory / 'data'}",
             f"--out_dir={self.out}", *options],
            cwd=REPOSITORY, capture_output=True, text=True, timeout=300,
        )

    def test_trains_until_its_time_budget_is_spent_and_writes_a_summary(self):
        done = self.train_best("--time_budget=1")
        self.assertEqual(done.returncode, 0, done.stderr[-800:])
        summary = json.loads((self.out / "summary.json").read_text())
        self.assertGreaterEqual(summary["training_seconds"], 1)
        self.assertGreater(summary["updates"], 10)
        self.assertEqual(summary["tokens"], summary["updates"] * 2 * 16)
        self.assertEqual(self.saved()["iter_num"], summary["updates"])

    def test_refuses_to_average_weights_with_a_time_budget(self):
        done = self.train_best("--time_budget=1", "--average_last=2")
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("time_budget=0", done.stderr)


class TrainingPartsTest(TinyRun, unittest.TestCase):
    def test_trains_with_document_attention_stem_triples_xielu_and_cautious_adamw(self):
        done = self.train("--document_attention=True", "--stem=True", "--bigram_table=True", "--trigram_table=True",
                          "--bigram_rows_per_token=4", "--xielu=True", "--muon_cautious_adamw=True",
                          "--muon_warmdown_share=0.75")
        self.assertEqual(done.returncode, 0, done.stderr[-800:])
        model_args = self.saved()["model_args"]
        self.assertEqual(model_args["document_bos"], 0)
        self.assertTrue(model_args["stem"] and model_args["trigram_table"] and model_args["xielu"])
        self.assertEqual(model_args["bigram_rows_per_token"], 4)

    def test_lets_attention_cross_documents_unless_asked(self):
        done = self.train()
        self.assertEqual(done.returncode, 0, done.stderr[-800:])
        self.assertEqual(self.saved()["model_args"]["document_bos"], -1)


class MixedWeightsTest(TinyRun, unittest.TestCase):
    def test_writes_the_last_weights_mixed_with_the_average_of_the_last_few(self):
        done = self.train("--average_last=3", "--average_every=2")
        self.assertEqual(done.returncode, 0, done.stderr[-800:])
        last, mixed = self.saved(), self.saved("mixed")
        self.assertEqual(mixed["model"].keys(), last["model"].keys())
        self.assertEqual(mixed["model_args"], last["model_args"])
        self.assertFalse(all(torch.equal(mixed["model"][name], value) for name, value in last["model"].items()))

    def test_mixes_the_same_last_weights_that_it_saves(self):
        done = self.train("--average_last=3", "--average_every=2", "--average_share=0.0")
        self.assertEqual(done.returncode, 0, done.stderr[-800:])
        last, mixed = self.saved(), self.saved("mixed")
        self.assertTrue(all(torch.equal(mixed["model"][name], value) for name, value in last["model"].items()))

    def test_refuses_to_average_weights_from_before_training(self):
        done = self.train("--average_last=8", "--average_every=32")
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("after the first update", done.stderr)


if __name__ == "__main__":
    unittest.main()
