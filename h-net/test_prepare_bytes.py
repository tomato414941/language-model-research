"""Checks of prepare_bytes.py on small corpora written as language-model's recipe writes them."""

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "h-net")]

import prepare  # noqa: E402
import prepare_bytes as module  # noqa: E402

JAPANESE = [f"文書{number}は犬について書かれていて、{number * 7}語の長さです。" for number in range(40)]
ENGLISH = [f"Document {number} is about dogs, and it is {number * 7} words long." for number in range(40)]


class PrepareBytesTest(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory)
        self.recipe = prepare.train_tokenizer(iter(JAPANESE + ENGLISH), 400)
        self.recipe.add_special_tokens([module.EOS])
        path = self.directory / "recipe.json"
        self.recipe.save(str(path))
        data = path.read_bytes()
        tokenizer = {"url": path.as_uri(), "sha256": hashlib.sha256(data).hexdigest()}
        self.sources = {
            "tokenizer": tokenizer,
            "stable": {"train_ja": self.packed(JAPANESE[:30], "s_ja"), "train_en": self.packed(ENGLISH[:30], "s_en"),
                       "validation_ja": self.packed(JAPANESE[30:], "v_ja"),
                       "validation_en": self.packed(ENGLISH[30:], "v_en")},
            "decay": {"train_ja": self.packed(JAPANESE[:20], "d_ja"), "train_en": self.packed(ENGLISH[:20], "d_en")},
        }
        self.data = self.directory / "data"
        self.manifest = module.prepare(self.sources, self.data)

    def packed(self, texts, name):
        bos, eos = self.recipe.token_to_id(module.BOS), self.recipe.token_to_id(module.EOS)
        tokens = [t for text in texts for t in (bos, *self.recipe.encode(text, add_special_tokens=False).ids, eos)]
        data = np.array(tokens, dtype=np.uint16).tobytes()
        path = self.directory / f"{name}.bin"
        path.write_bytes(data)
        return {"url": path.as_uri(), "sha256": hashlib.sha256(data).hexdigest()}

    def byte_documents(self, path):
        tokens = np.fromfile(path, dtype=np.uint16)
        heads = np.append(np.flatnonzero(tokens == module.BYTE_BOS), len(tokens))
        return [bytes(tokens[a + 1 : b].astype(np.uint8)).decode("utf-8") for a, b in zip(heads[:-1], heads[1:])]

    def token_documents(self, directory):
        tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        tokens = np.fromfile(directory / "train.bin", dtype=np.uint16)
        bos = tokenizer.token_to_id(module.BOS)
        heads = np.append(np.flatnonzero(tokens == bos), len(tokens))
        return [tokenizer.decode(tokens[a + 1 : b].tolist()) for a, b in zip(heads[:-1], heads[1:])]

    def test_writes_the_same_documents_in_the_same_order_as_bytes_and_as_tokens(self):
        for phase, count in (("stable", 30), ("decay", 20)):
            as_bytes = self.byte_documents(self.data / f"hnet-bytes-{phase}" / "train.bin")
            self.assertEqual(as_bytes, self.token_documents(self.data / f"hnet-bpe-{phase}"))
            self.assertCountEqual(as_bytes, JAPANESE[:count] + ENGLISH[:count])

    def test_mixes_the_languages(self):
        as_bytes = self.byte_documents(self.data / "hnet-bytes-stable" / "train.bin")
        self.assertNotEqual(as_bytes, JAPANESE[:30] + ENGLISH[:30])
        self.assertEqual(self.manifest["datasets"]["stable"]["documents"], {"ja": 30, "en": 30})

    def test_keeps_the_validation_texts_by_language(self):
        rows = [json.loads(line) for line in (self.data / "hnet-validation.jsonl").read_text().splitlines()]
        self.assertEqual([row["text"] for row in rows if row["language"] == "ja"], JAPANESE[30:])
        self.assertEqual([row["text"] for row in rows if row["language"] == "en"], ENGLISH[30:])

    def test_marks_each_byte_document_for_rows_py(self):
        directory = self.data / "hnet-bytes-stable"
        added = json.loads((directory / "tokenizer.json").read_text())["added_tokens"]
        self.assertEqual(added, [{"id": module.BYTE_BOS, "content": module.BOS}])
        self.assertEqual(json.loads((directory / "rows.json").read_text())["block_size"] % 256, 0)


if __name__ == "__main__":
    unittest.main()
