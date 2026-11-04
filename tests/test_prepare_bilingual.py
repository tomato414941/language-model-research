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
import prepare_bilingual as module

ENGLISH = [f"Document {number} is about dogs, and it is {number * 7} words long." for number in range(40)]
JAPANESE = [f"文書{number}は猫について書かれていて、長さは{number * 3}文字です。" for number in range(40)]


class PrepareBilingualTest(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory)
        # The recipe's tokenizer, whose marks stand at numbers of its own.
        self.recipe = prepare.train_tokenizer(iter(ENGLISH + JAPANESE), 400)
        self.recipe.add_special_tokens([module.EOS])
        self.recipe_path = self.directory / "recipe-tokenizer.json"
        self.recipe.save(str(self.recipe_path))

    def packed(self, texts, name):
        """A packed file of the documents, as the recipe writes them, and the source that names it."""
        bos, eos = self.recipe.token_to_id(module.BOS), self.recipe.token_to_id(module.EOS)
        tokens = [token for text in texts for token in (bos, *self.recipe.encode(text, add_special_tokens=False).ids, eos)]
        data = np.array(tokens, dtype=np.uint16).tobytes()
        path = self.directory / name
        path.write_bytes(data)
        return {"url": path.as_uri(), "sha256": hashlib.sha256(data).hexdigest()}

    def sources(self):
        data = self.recipe_path.read_bytes()
        return {
            "stable": {"train_ja": self.packed(JAPANESE[:20], "s-ja.bin"), "train_en": self.packed(ENGLISH[:20], "s-en.bin"),
                       "validation_ja": self.packed(JAPANESE[38:], "v-ja.bin"),
                       "validation_en": self.packed(ENGLISH[38:], "v-en.bin")},
            "decay": {"train_ja": self.packed(JAPANESE[20:38], "d-ja.bin"), "train_en": self.packed(ENGLISH[20:38], "d-en.bin")},
            "tokenizer": {"url": self.recipe_path.as_uri(), "sha256": hashlib.sha256(data).hexdigest()},
        }

    def prepared(self, sources=None):
        output = self.directory / "data" / "bilingual"
        manifest = module.prepare_bilingual(output, sources or self.sources(), table_share=0.25, tokenizer_bytes=2000,
                                            vocabulary=500)
        return output, manifest

    def documents(self, path):
        tokenizer = Tokenizer.from_file(str(path.parent / "tokenizer.json"))
        tokens = np.fromfile(path, dtype=np.uint16)
        heads = np.flatnonzero(tokens == tokenizer.token_to_id(module.BOS))
        ends = np.append(heads[1:], len(tokens))
        return [tokenizer.decode(tokens[head + 1 : end].tolist()) for head, end in zip(heads, ends)]

    def test_puts_each_document_of_both_languages_in_training_or_the_tables_text_once(self):
        output, _ = self.prepared()
        stable, decay = self.documents(output / "stable" / "train.bin"), self.documents(output / "decay" / "train.bin")
        table = self.documents(output / "table" / "train.bin")
        self.assertCountEqual(stable + decay + table, JAPANESE[:38] + ENGLISH[:38])

    def test_keeps_each_phase_to_its_own_documents(self):
        output, _ = self.prepared()
        self.assertTrue(set(self.documents(output / "stable" / "train.bin")) <= set(JAPANESE[:20] + ENGLISH[:20]))
        self.assertTrue(set(self.documents(output / "decay" / "train.bin")) <= set(JAPANESE[20:38] + ENGLISH[20:38]))

    def test_gives_the_table_about_its_share_of_each_phase(self):
        output, manifest = self.prepared()
        self.assertEqual(len(self.documents(output / "table" / "train.bin")), 10 + 9)
        self.assertEqual(manifest["tokens"]["stable/table"] + manifest["tokens"]["decay/table"],
                         len(np.fromfile(output / "table" / "train.bin", dtype=np.uint16)))

    def test_validates_on_both_languages_in_both_phases(self):
        output, _ = self.prepared()
        expected = JAPANESE[38:] + ENGLISH[38:]
        self.assertCountEqual(self.documents(output / "stable" / "val.bin"), expected)
        self.assertCountEqual(self.documents(output / "decay" / "val.bin"), expected)

    def test_gives_every_part_one_new_tokenizer_and_nanogpt_its_vocabulary(self):
        output, manifest = self.prepared()
        digests = {hashlib.sha256((output / part / "tokenizer.json").read_bytes()).hexdigest()
                   for part in ("stable", "decay", "table")}
        self.assertEqual(len(digests), 1)
        self.assertNotEqual(digests, {hashlib.sha256(self.recipe_path.read_bytes()).hexdigest()})
        for phase in ("stable", "decay"):
            with (output / phase / "meta.pkl").open("rb") as meta:
                self.assertEqual(pickle.load(meta), {"vocab_size": manifest["vocabulary"]})
        # The few test documents give fewer merges than the vocabulary asked for.
        tokenizer = Tokenizer.from_file(str(output / "stable" / "tokenizer.json"))
        self.assertEqual(manifest["vocabulary"], tokenizer.get_vocab_size())
        self.assertLessEqual(manifest["vocabulary"], 500)

    def test_records_what_it_read_and_wrote(self):
        output, manifest = self.prepared()
        self.assertEqual(json.loads((output / "manifest.json").read_text()), manifest)
        self.assertEqual(manifest["documents"]["stable/train_ja"], 20)
        self.assertEqual(manifest["sha256"]["stable/train.bin"], prepare.file_digest(output / "stable" / "train.bin"))

    def test_refuses_a_file_that_is_not_the_one_its_source_names(self):
        sources = self.sources()
        sources["decay"]["train_en"]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            self.prepared(sources)
        self.assertFalse((self.directory / "data" / "bilingual").exists())

    def test_samples_the_tokenizers_text_from_across_each_file(self):
        path = self.directory / "s-en.bin"
        self.packed(ENGLISH, path.name)
        taken = list(module.sample([path], self.recipe, limit=200, seed=1))
        self.assertLess(len(taken), len(ENGLISH))
        self.assertNotEqual(taken, ENGLISH[: len(taken)])
        self.assertTrue(set(taken) <= set(ENGLISH))


if __name__ == "__main__":
    unittest.main()
