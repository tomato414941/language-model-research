import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

if importlib.util.find_spec("numpy") is None:
    raise unittest.SkipTest("Install the dependencies")

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from rows import document_heads, whole_row_heads


class DocumentHeadsTest(unittest.TestCase):
    TOKENS = [7, 5, 6, 7, 8, 7, 9, 7, 1, 2, 3, 4]

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.tokens = self.directory / "train.bin"
        self.tokenizer = self.directory / "tokenizer.json"
        np.array(self.TOKENS, dtype=np.uint16).tofile(self.tokens)
        self.tokenizer.write_text(json.dumps({"added_tokens": [{"id": 7, "content": "<|bos|>"}]}), encoding="utf-8")

    def test_finds_the_head_of_every_document(self):
        self.assertEqual(document_heads(self.tokens, self.tokenizer).tolist(), [0, 3, 5, 7])

    def test_reads_the_heads_back_from_the_file_beside_the_tokens(self):
        document_heads(self.tokens, self.tokenizer)
        self.tokens.unlink()
        self.assertEqual(document_heads(self.tokens, self.tokenizer).tolist(), [0, 3, 5, 7])

    def test_leaves_nothing_unfinished_beside_the_tokens(self):
        document_heads(self.tokens, self.tokenizer)
        self.assertEqual(sorted(path.name for path in self.directory.iterdir()),
                         ["tokenizer.json", "train.bin", "train.heads.npy"])

    def test_keeps_only_the_heads_with_room_for_a_row_and_its_targets(self):
        heads = np.array([0, 3, 5, 7])
        self.assertEqual(whole_row_heads(heads, 12, 4).tolist(), [0, 3, 5, 7])
        self.assertEqual(whole_row_heads(heads, 12, 5).tolist(), [0, 3, 5])
        self.assertEqual(whole_row_heads(heads, 12, 8).tolist(), [0, 3])


if __name__ == "__main__":
    unittest.main()
