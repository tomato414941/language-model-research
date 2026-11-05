"""Checks of evaluate_hnet.py's validation BPB on a small dataset written as prepare.py writes one."""

import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "h-net")]

import prepare  # noqa: E402
from evaluate_hnet import validation_bits  # noqa: E402
from hnet_model import BOS  # noqa: E402
from loading import LanguageModel  # noqa: E402

TEXTS = [f"文書{n}は犬について。Document {n} is about dogs, {n * 7} words; 🐕 ñ\n" * 3 for n in range(60)]


def uniform_model(context, seen):
    """A model of bytes that gives every byte the same chance, and keeps the rows it reads."""

    def logits(rows):
        seen.append(rows.clone())
        return torch.zeros(*rows.shape, BOS + 1)

    return LanguageModel(logits=logits, encode=lambda text: list(text.encode("utf-8")), context=context, prefix=BOS,
                         batch_tokens=4096)


class ValidationBitsTest(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory)
        tokenizer = prepare.train_tokenizer(iter(TEXTS), 400)
        tokenizer.save(str(self.directory / "tokenizer.json"))
        prepare.write_tokens(iter(TEXTS), tokenizer, self.directory / "val.bin")

    def test_gives_a_uniform_guess_about_log2_of_its_choices_per_byte(self):
        # Not exactly: a row that ends inside a character decodes to fewer or more bytes than its tokens hold.
        bits = validation_bits(uniform_model(3840, []), str(self.directory))
        self.assertAlmostEqual(bits / math.log2(BOS + 1), 1, delta=0.01)

    def test_reads_rows_longer_than_the_context_in_pieces_within_it(self):
        seen = []
        bits = validation_bits(uniform_model(100, seen), str(self.directory))
        # The same bits as reading each row whole, up to the rounding of sums in 32 bits; a byte read twice or never
        # would move them by about one part in ten thousand.
        whole = validation_bits(uniform_model(3840, []), str(self.directory))
        self.assertAlmostEqual(bits, whole, delta=whole * 1e-5)
        self.assertLessEqual(max(rows.size(1) for rows in seen), 100)
        tokens = np.fromfile(self.directory / "val.bin", dtype=np.uint16)
        self.assertGreater(len(tokens), 1024)


if __name__ == "__main__":
    unittest.main()
