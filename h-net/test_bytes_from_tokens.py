"""Checks of bytes_from_tokens.py on a small dataset written as prepare.py writes one."""

import pickle
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "h-net")]

import bytes_from_tokens as module  # noqa: E402
import prepare  # noqa: E402
import rows  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

TEXTS = [f"文書{n}は犬について。Document {n} is about dogs, {n * 7} words; 🐕 ñ\n" * 3 for n in range(50)]


class BytesFromTokensTest(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory)
        source = self.directory / "tokens"
        source.mkdir()
        tokenizer = prepare.train_tokenizer(iter(TEXTS), 400)
        tokenizer.save(str(source / "tokenizer.json"))
        prepare.write_tokens(iter(TEXTS[:40]), tokenizer, source / "train.bin")
        prepare.write_tokens(iter(TEXTS[40:]), tokenizer, source / "val.bin")
        self.output = self.directory / "bytes"
        self.block_size, self.counts = module.bytes_from_tokens(source, self.output)

    def documents(self, path):
        values = np.fromfile(path, dtype=np.uint16)
        heads = np.append(np.flatnonzero(values == module.BYTE_BOS), len(values))
        return [bytes(values[a + 1 : b].astype(np.uint8)).decode("utf-8") for a, b in zip(heads[:-1], heads[1:])]

    def test_writes_each_document_as_its_utf8_bytes_in_the_same_order(self):
        self.assertEqual(self.documents(self.output / "train.bin"), TEXTS[:40])
        self.assertEqual(self.documents(self.output / "val.bin"), TEXTS[40:])

    def test_gives_train_py_and_rows_py_what_they_read(self):
        with (self.output / "meta.pkl").open("rb") as meta:
            self.assertEqual(pickle.load(meta), {"vocab_size": module.BYTE_BOS + 1})
        self.assertEqual(rows.bos_token(self.output / "tokenizer.json"), module.BYTE_BOS)

    def test_writes_a_tokenizer_whose_tokens_are_the_text_s_utf8_bytes(self):
        tokenizer = Tokenizer.from_file(str(self.output / "tokenizer.json"))
        text = TEXTS[3]
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        self.assertEqual(ids, list(text.encode("utf-8")))
        self.assertEqual(tokenizer.decode(ids), text)
        self.assertEqual(tokenizer.get_vocab_size(), module.BYTE_BOS + 1)

    def test_counts_one_byte_for_each_byte_token_and_none_for_bos(self):
        from frame import token_bytes

        sizes = token_bytes(self.output / "tokenizer.json")
        self.assertEqual(sizes.tolist(), [1] * 256 + [0])

    def test_sizes_a_row_to_hold_the_text_of_1024_tokens(self):
        train = self.counts["train"]
        per_token = (train["bytes"] - train["documents"]) / (train["tokens"] - train["documents"])
        self.assertEqual(self.block_size % 256, 0)
        self.assertLessEqual(abs(self.block_size - 1024 * per_token), 128)


if __name__ == "__main__":
    unittest.main()
