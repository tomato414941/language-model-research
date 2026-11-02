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

REPOSITORY = Path(__file__).parent.parent
VOCABULARY = 16


class DecayDatasetTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def dataset(self, name, last_token=VOCABULARY - 1):
        """A dataset of short documents, with its validation text, which training can only read below the vocabulary."""
        path = self.directory / name
        path.mkdir()
        tokens = np.tile(np.array([0, 1, 2, 3, 4, last_token], dtype=np.uint16), 200)
        tokens.tofile(path / "train.bin")
        np.tile(np.arange(VOCABULARY, dtype=np.uint16), 20).tofile(path / "val.bin")
        (path / "tokenizer.json").write_text(json.dumps({"added_tokens": [{"id": 0, "content": "<|bos|>"}]}))
        with (path / "meta.pkl").open("wb") as meta:
            pickle.dump({"vocab_size": VOCABULARY}, meta)
        return path

    def train(self, first, later, updates=8):
        return subprocess.run(
            [sys.executable, "-u", "train.py", "configs/bundle2.py", "--device=cpu", "--dtype=float32", "--compile=False",
             "--n_layer=1", "--n_embd=32", "--n_head=1", "--block_size=8", "--batch_size=2", "--gradient_accumulation_steps=1",
             f"--max_iters={updates}", f"--lr_decay_iters={updates}", "--eval_interval=1000", "--eval_iters=1", "--log_interval=1",
             f"--dataset={first}", f"--decay_dataset={later}", f"--out_dir={self.directory / 'out'}"],
            cwd=REPOSITORY, capture_output=True, text=True, timeout=300,
        )

    def test_reads_the_decay_dataset_from_where_the_learning_rate_begins_to_fall(self):
        done = self.train(self.dataset("first"), self.dataset("later"))
        self.assertEqual(done.returncode, 0, done.stderr[-800:])
        self.assertIn(f"iter 5: reading {self.directory / 'later'}", done.stdout)

    def test_reads_the_first_dataset_until_then(self):
        # A token beyond the vocabulary stops training at the first row that holds it.
        done = self.train(self.dataset("first"), self.dataset("later", last_token=VOCABULARY + 100))
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("iter 3: loss", done.stdout)
        self.assertNotIn("iter 6: loss", done.stdout)


if __name__ == "__main__":
    unittest.main()
