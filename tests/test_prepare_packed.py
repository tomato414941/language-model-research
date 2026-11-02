import hashlib
import importlib.util
import json
import pickle
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

if any(importlib.util.find_spec(name) is None for name in ("numpy", "pyarrow", "tokenizers")):
    raise unittest.SkipTest("Install the dependencies")

import numpy as np
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

import prepare
import prepare_packed as module

TEXTS = [f"Document {number} is about dogs, and it is {number * 7} words long." for number in range(60)]


class PreparePackedTest(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory)
        self.tokenizer_path = self.directory / "tokenizer.json"
        self.tokenizer = prepare.train_tokenizer(iter(TEXTS), 300)
        self.tokenizer.save(str(self.tokenizer_path))
        # Another tokenizer, as a recipe's would be, whose marks stand at other numbers.
        self.recipe_path = self.directory / "recipe-tokenizer.json"
        self.recipe = prepare.train_tokenizer(iter(reversed(TEXTS)), 280)
        self.recipe.add_special_tokens([module.EOS])
        self.recipe.save(str(self.recipe_path))

    def packed(self, texts, name, cut=0):
        """A packed file of the documents, as the recipe writes them, and the source that names it."""
        bos, eos = self.recipe.token_to_id(module.BOS), self.recipe.token_to_id(module.EOS)
        tokens = [token for text in texts for token in (bos, *self.recipe.encode(text, add_special_tokens=False).ids, eos)]
        data = np.array(tokens[: len(tokens) - cut], dtype=np.uint16).tobytes()
        path = self.directory / name
        path.write_bytes(data)
        return {"url": path.as_uri(), "sha256": hashlib.sha256(data).hexdigest()}

    def prepared(self, train=TEXTS, val=TEXTS[:5]):
        output = self.directory / "data" / "corpus"
        sources = {"train": self.packed(train, "train_en.bin"), "val": self.packed(val, "validation_en.bin")}
        return output, module.prepare_packed(output, self.tokenizer_path, self.recipe_path, sources), sources

    def documents(self, path):
        """The texts of a prepared file's documents, in its order."""
        tokens = np.fromfile(path, dtype=np.uint16)
        heads = np.flatnonzero(tokens == self.tokenizer.token_to_id(module.BOS))
        ends = np.append(heads[1:], len(tokens))
        return [self.tokenizer.decode(tokens[head + 1 : end].tolist()) for head, end in zip(heads, ends)]

    def test_keeps_every_document_in_the_prepared_tokenizer(self):
        output, _, _ = self.prepared()
        self.assertCountEqual(self.documents(output / "train.bin"), TEXTS)
        self.assertCountEqual(self.documents(output / "val.bin"), TEXTS[:5])

    def test_shuffles_the_documents(self):
        output, _, _ = self.prepared()
        self.assertNotEqual(self.documents(output / "train.bin"), TEXTS)

    def test_keeps_a_document_whose_text_spells_a_mark_whole(self):
        spelled = f"It ends with {module.EOS} in its text and goes on."
        output, manifest, _ = self.prepared(train=[TEXTS[0], spelled, TEXTS[1]])
        self.assertEqual(manifest["documents"]["train"], 3)
        self.assertEqual(len(self.documents(output / "train.bin")), 3)

    def test_counts_what_it_wrote(self):
        output, manifest, _ = self.prepared()
        self.assertEqual(manifest["documents"], {"train": len(TEXTS), "val": 5})
        self.assertEqual(manifest["tokens"]["train"], len(np.fromfile(output / "train.bin", dtype=np.uint16)))

    def test_gives_nanogpt_its_vocabulary_and_tokenizer(self):
        output, manifest, _ = self.prepared()
        with (output / "meta.pkl").open("rb") as meta:
            self.assertEqual(pickle.load(meta)["vocab_size"], manifest["vocabulary"])
        self.assertEqual(Tokenizer.from_file(str(output / "tokenizer.json")).get_vocab_size(), manifest["vocabulary"])

    def test_records_checksums_and_leaves_out_the_urls(self):
        output, manifest, sources = self.prepared()
        written = (output / "manifest.json").read_text()
        self.assertEqual(json.loads(written)["sha256"]["train.bin"], prepare.file_digest(output / "train.bin"))
        self.assertEqual(manifest["packed_sha256"]["train"], sources["train"]["sha256"])
        self.assertNotIn(sources["train"]["url"], written)

    def test_refuses_a_file_other_than_the_one_named(self):
        sources = {"train": self.packed(TEXTS, "train_en.bin"), "val": self.packed(TEXTS[:5], "validation_en.bin")}
        sources["val"]["sha256"] = "0" * 64
        output = self.directory / "data" / "corpus"
        with self.assertRaisesRegex(ValueError, "val file"):
            module.prepare_packed(output, self.tokenizer_path, self.recipe_path, sources)
        self.assertEqual(list(output.parent.iterdir()), [])

    def test_refuses_a_file_that_stops_inside_a_document(self):
        sources = {"train": self.packed(TEXTS, "train_en.bin", cut=2), "val": self.packed(TEXTS[:5], "validation_en.bin")}
        with self.assertRaisesRegex(ValueError, "inside a document"):
            module.prepare_packed(self.directory / "data" / "corpus", self.tokenizer_path, self.recipe_path, sources)

    def test_refuses_to_replace_a_directory(self):
        output, _, sources = self.prepared()
        with self.assertRaisesRegex(ValueError, "already exists"):
            module.prepare_packed(output, self.tokenizer_path, self.recipe_path, sources)


if __name__ == "__main__":
    unittest.main()
