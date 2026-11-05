import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path

if any(importlib.util.find_spec(name) is None for name in ("numpy", "torch", "tokenizers")):
    raise unittest.SkipTest("Install the dependencies")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import frame
from model import GPT, GPTConfig
from prepare import BOS, train_tokenizer

TEXTS = ["The cat sat on the mat. Ünïcödé text, with numbers 12345.", "Another document; café au lait."] * 20


class FrameTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.tokenizer_path = self.directory / "tokenizer.json"
        self.tokenizer = train_tokenizer(iter(TEXTS), 300)
        self.tokenizer.save(str(self.tokenizer_path))
        self.bos = self.tokenizer.token_to_id(BOS)


class CoreDecisionTest(unittest.TestCase):
    def test_adopts_only_a_higher_score(self):
        self.assertTrue(frame.core_improves(0.1000000000000001, 0.1))
        self.assertFalse(frame.core_improves(0.1, 0.1))
        self.assertFalse(frame.core_improves(0.09, 0.1))

    def test_refuses_to_decide_on_a_score_that_is_not_finite(self):
        with self.assertRaises(ValueError):
            frame.core_improves(math.nan, 0.1)

    def test_refuses_assets_that_are_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                frame.verify_core_assets(directory)


class TokenizerTest(FrameTest):
    def test_counts_the_bytes_each_token_stands_for_and_none_for_bos(self):
        sizes = frame.token_bytes(self.tokenizer_path)
        for text in TEXTS[:2]:
            ids = self.tokenizer.encode(text, add_special_tokens=False).ids
            self.assertEqual(int(sizes[ids].sum()), len(text.encode("utf-8")))
        self.assertEqual(sizes[self.bos], 0)

    def test_encodes_a_list_of_texts_with_a_token_put_first_as_nanochat_asks(self):
        tokenizer = frame.CoreTokenizer(self.tokenizer_path)
        self.assertEqual(tokenizer.get_bos_token_id(), self.bos)
        encoded = tokenizer(TEXTS[:2], prepend=self.bos)
        self.assertEqual(len(encoded), 2)
        for text, ids in zip(TEXTS[:2], encoded):
            self.assertEqual(ids, [self.bos, *self.tokenizer.encode(text, add_special_tokens=False).ids])
        self.assertEqual(tokenizer(TEXTS[0]), self.tokenizer.encode(TEXTS[0], add_special_tokens=False).ids)


def small_model(vocabulary):
    torch.manual_seed(0)
    return GPT(GPTConfig(block_size=8, vocab_size=vocabulary, n_layer=1, n_head=2, n_embd=16, bias=False))


class ModelTest(FrameTest):
    def test_gives_nanochat_the_logits_of_every_position_and_the_context(self):
        network = small_model(300)
        model = frame.CoreModel(network)
        tokens = torch.randint(0, 300, (3, 6), generator=torch.Generator().manual_seed(1))
        self.assertEqual(model(tokens).shape, (3, 6, 300))
        self.assertEqual(model.max_seq_len, 8)

    def test_reads_rows_from_document_heads_and_counts_the_bytes_of_their_text(self):
        network = small_model(300)
        documents = [self.tokenizer.encode(text, add_special_tokens=False).ids for text in TEXTS[:4]]
        tokens = np.array([token for ids in documents for token in [self.bos, *ids]], dtype=np.uint16)
        validation = self.directory / "val.bin"
        tokens.tofile(validation)
        bpb = frame.evaluate_bpb(network, validation, self.tokenizer_path, "cpu", rows_per_pass=2)

        sizes = frame.token_bytes(self.tokenizer_path)
        heads, starts, free = np.flatnonzero(tokens == self.bos), [], 0
        for head in heads:
            if head >= free and head + 9 <= len(tokens):
                starts.append(head)
                free = head + 8
        rows = torch.from_numpy(np.stack([tokens[start:start + 9] for start in starts]).astype(np.int64))
        with torch.no_grad():
            logits, _ = network(rows[:, :-1].contiguous(), rows[:, 1:].contiguous())
        losses = torch.nn.functional.cross_entropy(logits.flatten(0, 1), rows[:, 1:].flatten(), reduction="none")
        counted = torch.from_numpy(sizes)[rows[:, 1:].flatten()]
        expected = (losses * (counted > 0)).sum().item() / (math.log(2) * counted.sum().item())
        self.assertGreater(len(starts), 1)
        self.assertAlmostEqual(bpb, expected, places=5)


if __name__ == "__main__":
    unittest.main()
