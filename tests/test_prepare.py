import importlib.util
import json
import pickle
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

if any(importlib.util.find_spec(name) is None for name in ("numpy", "pyarrow", "tokenizers")):
    raise unittest.SkipTest("Install the dependencies")

import numpy as np
import pyarrow as arrow
import pyarrow.parquet as parquet
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

import prepare as module


class PrepareTest(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory)
        self.requested = []
        patcher = patch.object(module, "TOKENIZER_CHARACTERS", 5000)
        patcher.start()
        self.addCleanup(patcher.stop)

    def download(self, index, directory):
        """A shard of short documents; the validation shard's are the only ones about cats."""
        self.requested.append(index)
        subject = "cats" if index == module.LAST_SHARD else "dogs"
        texts = [f"Document {number} of shard {index} is about {subject}." for number in range(50)]
        path = Path(directory) / f"shard_{index:05d}.parquet"
        parquet.write_table(arrow.table({"text": texts}), path)
        return path

    def prepared(self, train_tokens=2000, vocabulary=300):
        output = self.directory / "data" / "fineweb"
        manifest = module.prepare(output, train_tokens, vocabulary, self.download)
        return output, manifest

    def read(self, path):
        return np.fromfile(path, dtype=np.uint16)

    def test_writes_exactly_the_tokens_training_asks_for(self):
        output, manifest = self.prepared(train_tokens=2000)
        self.assertEqual(len(self.read(output / "train.bin")), 2000)
        self.assertEqual(manifest["tokens"]["train"], 2000)

    def test_begins_every_document_with_bos(self):
        output, _ = self.prepared(train_tokens=2000)
        tokenizer = Tokenizer.from_file(str(output / "tokenizer.json"))
        tokens = self.read(output / "train.bin")
        self.assertEqual(tokens[0], tokenizer.token_to_id(module.BOS))
        self.assertGreater((tokens == tokenizer.token_to_id(module.BOS)).sum(), 10)

    def test_holds_out_the_last_shard_for_validation(self):
        output, _ = self.prepared(train_tokens=2000)
        tokenizer = Tokenizer.from_file(str(output / "tokenizer.json"))
        self.assertIn("cats", tokenizer.decode(self.read(output / "val.bin").tolist()))
        self.assertNotIn("cats", tokenizer.decode(self.read(output / "train.bin").tolist()))

    def test_gives_nanogpt_the_vocabulary_size(self):
        output, manifest = self.prepared(vocabulary=300)
        with (output / "meta.pkl").open("rb") as meta:
            self.assertEqual(pickle.load(meta)["vocab_size"], manifest["vocabulary"])
        self.assertLess(self.read(output / "train.bin").max(), manifest["vocabulary"])

    def test_restores_the_text_it_encoded(self):
        output, _ = self.prepared()
        tokenizer = Tokenizer.from_file(str(output / "tokenizer.json"))
        text = "Document 12 of shard 0 is about dogs."
        self.assertEqual(tokenizer.decode(tokenizer.encode(text).ids), text)

    def test_records_where_the_data_came_from_and_its_checksums(self):
        output, _ = self.prepared()
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["revision"], module.REVISION)
        self.assertEqual(manifest["sha256"]["train.bin"], module.file_digest(output / "train.bin"))

    def test_leaves_no_shards_behind(self):
        output, _ = self.prepared()
        self.assertFalse((output / "shards").exists())

    def test_refuses_to_replace_a_directory(self):
        output, _ = self.prepared()
        with self.assertRaisesRegex(ValueError, "already exists"):
            module.prepare(output, 10, 300, self.download)


if __name__ == "__main__":
    unittest.main()
