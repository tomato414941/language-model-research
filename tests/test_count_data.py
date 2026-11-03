import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

if any(importlib.util.find_spec(name) is None for name in ("numpy", "pyarrow", "tokenizers", "ahocorasick")):
    raise unittest.SkipTest("Install the dependencies")

import numpy as np
import pyarrow as arrow
import pyarrow.parquet as parquet
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

import count_data as module
import prepare

QUESTION = "Which gas do plants take in from the air to make their food during photosynthesis?"


class CountDataTest(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory)
        texts = [f"Document {number} is about dogs and their walks in the park." for number in range(200)]
        tokenizer = prepare.train_tokenizer(iter(texts), 300)
        self.tokenizer_path = self.directory / "tokenizer.json"
        tokenizer.save(str(self.tokenizer_path))
        self.questions_path = self.directory / "questions.jsonl"
        self.questions_path.write_text(json.dumps({"context": f"Question: {QUESTION}\nAnswer:"}) + "\n")

    def download(self, index, directory):
        """A file of short documents; the first file's third one quotes the question."""
        texts = [f"Document {number} of file {index} is about dogs." for number in range(50)]
        if index == 0:
            texts[2] = f"A teacher once asked: {QUESTION.upper()} Nobody knew."
        path = Path(directory) / f"{index:03d}_00000.parquet"
        parquet.write_table(arrow.table({"text": texts}), path)
        return path

    def prepared(self, tokens):
        output = self.directory / "count-text"
        manifest = module.prepare(output, self.tokenizer_path, self.questions_path, tokens, self.download, workers=2)
        return output, manifest

    def test_writes_exactly_the_tokens_asked_for(self):
        output, manifest = self.prepared(3000)
        self.assertEqual(len(np.fromfile(output / "train.bin", dtype=np.uint16)), 3000)
        self.assertEqual(manifest["tokens"], 3000)

    def test_begins_every_document_with_bos(self):
        output, _ = self.prepared(3000)
        bos = Tokenizer.from_file(str(self.tokenizer_path)).token_to_id(module.BOS)
        tokens = np.fromfile(output / "train.bin", dtype=np.uint16)
        self.assertEqual(tokens[0], bos)
        self.assertGreater((tokens == bos).sum(), 50)

    def test_leaves_out_the_documents_that_hold_a_question(self):
        output, manifest = self.prepared(3000)
        tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        text = tokenizer.decode(np.fromfile(output / "train.bin", dtype=np.uint16).tolist())
        self.assertNotIn("teacher", text)
        self.assertIn("Document 3 of file 0", text)
        self.assertEqual(manifest["documents_with_a_question"], 1)

    def test_finds_a_question_by_its_middle_whatever_the_case_and_punctuation(self):
        key = module.question_key(f"Question: {QUESTION}\nAnswer:")
        self.assertEqual(len(key), module.WIDTH)
        self.assertIn(key, module.normalized(QUESTION.upper().replace(" ", "  ")))

    def test_cannot_find_a_question_too_short_to_tell_from_other_text(self):
        self.assertIsNone(module.question_key("Question: What is 2 + 2?\nAnswer:"))

    def test_refuses_to_write_over_a_directory(self):
        (self.directory / "count-text").mkdir()
        with self.assertRaises(ValueError):
            self.prepared(3000)


if __name__ == "__main__":
    unittest.main()
