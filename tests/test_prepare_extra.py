import gzip
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

if any(importlib.util.find_spec(name) is None for name in ("numpy", "pyarrow", "tokenizers")):
    raise unittest.SkipTest("Install the dependencies")

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import prepare_extra
from prepare import BOS, train_tokenizer

DOCUMENTS = [["def add(a, b):\n    return a + b\n", "print('hi')\n"], ["x = 1\n"], ["never read\n"]]


class PrepareExtraTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.sources = self.directory / "sources"
        self.sources.mkdir()
        self.names = []
        for index, documents in enumerate(DOCUMENTS):
            name = f"stack-edu-{index:04d}.json.gz"
            with gzip.open(self.sources / name, "wt", encoding="utf-8") as lines:
                for text in documents:
                    lines.write(json.dumps({"id": str(index), "text": text}) + "\n")
            self.names.append(name)
        self.tokenizer = train_tokenizer(iter([text for documents in DOCUMENTS for text in documents] * 10), 280)
        self.tokenizer_path = self.directory / "tokenizer.json"
        self.tokenizer.save(str(self.tokenizer_path))

    def fetch(self, name, path, directory):
        target = Path(directory) / path
        shutil.copyfile(self.sources / path, target)
        return target

    def prepare(self, files):
        return prepare_extra.prepare("stackv2-edu", self.directory / "out", self.tokenizer_path, files,
                                     fetch=self.fetch, list_files=lambda name: self.names)

    def test_writes_the_first_files_documents_each_after_a_bos_token(self):
        manifest = self.prepare(2)
        bos = self.tokenizer.token_to_id(BOS)
        expected = []
        for text in DOCUMENTS[0] + DOCUMENTS[1]:
            expected += [bos, *self.tokenizer.encode(text, add_special_tokens=False).ids]
        written = np.fromfile(self.directory / "out" / "train.bin", dtype=np.uint16).tolist()
        self.assertEqual(written, expected)
        self.assertEqual(manifest["files"], self.names[:2])
        self.assertEqual(manifest["tokens"], len(expected))

    def test_reads_the_first_documents_of_every_file_as_they_arrive(self):
        manifest = prepare_extra.prepare("stackv2-edu", self.directory / "out", self.tokenizer_path,
                                         list_files=lambda name: self.names, documents_per_file=1,
                                         open_file=lambda name, path: open(self.sources / path, "rb"))
        bos = self.tokenizer.token_to_id(BOS)
        expected = []
        for documents in DOCUMENTS:
            expected += [bos, *self.tokenizer.encode(documents[0], add_special_tokens=False).ids]
        written = np.fromfile(self.directory / "out" / "train.bin", dtype=np.uint16).tolist()
        self.assertEqual(written, expected)
        self.assertEqual((manifest["files"], manifest["documents_per_file"]), (self.names, 1))

    def test_reads_the_files_in_the_order_it_is_given(self):
        manifest = prepare_extra.prepare("stackv2-edu", self.directory / "out", self.tokenizer_path, 1,
                                         fetch=self.fetch, list_files=lambda name: list(reversed(self.names)))
        self.assertEqual(manifest["files"], self.names[-1:])

    def test_keeps_the_tokenizer_and_vocabulary_beside_the_tokens(self):
        self.prepare(1)
        out = self.directory / "out"
        self.assertEqual((out / "tokenizer.json").read_bytes(), self.tokenizer_path.read_bytes())
        self.assertTrue((out / "meta.pkl").exists())

    def test_refuses_an_output_that_exists(self):
        (self.directory / "out").mkdir()
        with self.assertRaises(ValueError):
            self.prepare(1)

    def test_reads_the_text_of_parquet_files_too(self):
        import pyarrow as arrow
        import pyarrow.parquet as parquet

        path = self.directory / "train-00000.parquet"
        parquet.write_table(arrow.table({"text": ["one", "two"]}), path)
        self.assertEqual(list(prepare_extra.file_texts(path)), ["one", "two"])


if __name__ == "__main__":
    unittest.main()
