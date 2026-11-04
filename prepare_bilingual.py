"""Prepare what training reads from language-model's v6 corpora, Japanese and English, in a tokenizer trained on them.

The v6 recipe packs each corpus as 16-bit files of documents between BOS and EOS in its own tokenizer, one source
after another. Here each document is decoded with that tokenizer, a tokenizer of prepare.py's kind is trained on a
sample of both languages, as many bytes of each as v6's tokenizer took, and every document is encoded with it.
Each phase's documents, both languages together, are shuffled and split: most for training, the rest as text for
count_table.py, so that the COUNT table never counts the text that training reads. Validation is the stable
corpus's validation files of both languages.

The sources file is JSON, as v6_sources.py writes it: {"stable": {"train_ja": {"url": ..., "sha256": ...}, ...},
"decay": {...}, "tokenizer": {...}}. It is a file because its URLs grant access, and a command's arguments show
among a machine's processes.

Usage: python prepare_bilingual.py OUTPUT --sources FILE
It writes OUTPUT/stable and OUTPUT/decay, which nanoGPT reads as datasets, and OUTPUT/table/train.bin.
"""

import argparse
import hashlib
import json
import os
import pickle
import shutil
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from prepare import BOS, VOCABULARY, file_digest, train_tokenizer, write_tokens
from prepare_packed import EOS, SHUFFLE_SEED, decoded_texts, read_documents, read_tokens

PHASES = {"stable": ("train_ja", "train_en"), "decay": ("train_ja", "train_en")}
VALIDATION = ("validation_ja", "validation_en")
# v6's tokenizer took 64 MiB of text of each language.
TOKENIZER_BYTES_PER_LANGUAGE = 64 * 1024 * 1024
# The share of each phase's documents that goes to the COUNT table's text, about 2 billion tokens in all.
TABLE_SHARE = 0.45


def download(source, path):
    """Saves the source's file at the path, after checking it is the one the source names."""
    digest = hashlib.sha256()
    with urllib.request.urlopen(source["url"], timeout=120) as stream, open(path, "wb") as output:
        while chunk := stream.read(1 << 24):
            digest.update(chunk)
            output.write(chunk)
    if digest.hexdigest() != source["sha256"]:
        raise ValueError(f"{path.name} is not the file its source names")


def texts_of(path, recipe, tally):
    """The texts of a packed file's documents, counting them in the tally."""
    with open(path, "rb") as stream:
        documents = read_documents(read_tokens(stream, hashlib.sha256()), recipe.token_to_id(BOS),
                                   recipe.token_to_id(EOS))
        yield from decoded_texts(documents, recipe, tally)


def sample(paths, recipe, limit, seed):
    """Texts of documents taken at random from each file, up to `limit` bytes of each, one file after another.
    A file holds its sources one after another, so its first documents would all be of one source."""
    bos, eos = recipe.token_to_id(BOS), recipe.token_to_id(EOS)
    for path in paths:
        tokens = np.memmap(path, dtype=np.uint16, mode="r")
        heads = np.concatenate([[0], np.flatnonzero((tokens[:-1] == eos) & (tokens[1:] == bos)) + 1])
        ends = np.append(heads[1:], len(tokens))
        taken = 0
        for index in np.random.default_rng(seed).permutation(len(heads)).tolist():
            text = recipe.decode(tokens[heads[index] + 1 : ends[index] - 1].tolist(), skip_special_tokens=False)
            yield text
            taken += len(text.encode("utf-8"))
            if taken >= limit:
                break


def split_documents(source, first_path, rest_path, bos, seed, rest_share):
    """Writes the file's documents, each beginning at a BOS, in a random order: the first of them to one file and
    the last `rest_share` of them to the other. Returns the tokens written to each."""
    tokens = np.memmap(source, dtype=np.uint16, mode="r")
    heads = np.flatnonzero(tokens == bos)
    ends = np.append(heads[1:], len(tokens))
    order = np.random.default_rng(seed).permutation(len(heads))
    cut = len(order) - round(len(order) * rest_share)
    written = []
    for path, part in ((first_path, order[:cut]), (rest_path, order[cut:])):
        count = 0
        with open(path, "ab") as output:
            for index in part.tolist():
                output.write(tokens[heads[index] : ends[index]].tobytes())
                count += int(ends[index] - heads[index])
        written.append(count)
    return written


def prepare_bilingual(output, sources, seed=SHUFFLE_SEED, table_share=TABLE_SHARE,
                      tokenizer_bytes=TOKENIZER_BYTES_PER_LANGUAGE, vocabulary=VOCABULARY):
    output = Path(output)
    if output.exists():
        raise ValueError("output already exists; select a new directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        packed = staging / "packed"
        packed.mkdir()
        download(sources["tokenizer"], packed / "recipe-tokenizer.json")
        recipe = Tokenizer.from_file(str(packed / "recipe-tokenizer.json"))
        files = {}
        for phase, names in {**PHASES, "stable": PHASES["stable"] + VALIDATION}.items():
            for name in names:
                files[phase, name] = packed / f"{phase}-{name}.bin"
                download(sources[phase][name], files[phase, name])

        tokenizer = train_tokenizer(
            sample([files["stable", name] for name in PHASES["stable"]], recipe, tokenizer_bytes, seed), vocabulary)
        bos = tokenizer.token_to_id(BOS)
        table = staging / "table"
        table.mkdir()
        manifest = {"vocabulary": tokenizer.get_vocab_size(), "shuffle_seed": seed, "table_share": table_share,
                    "tokenizer_bytes_per_language": tokenizer_bytes,
                    "recipe_tokenizer_sha256": sources["tokenizer"]["sha256"],
                    "packed_sha256": {f"{phase}/{name}": sources[phase][name]["sha256"] for phase, name in files},
                    "documents": {}, "tokens": {}}
        for number, (phase, names) in enumerate(PHASES.items()):
            directory = staging / phase
            directory.mkdir()
            unshuffled = staging / f"{phase}.unshuffled.bin"
            with open(unshuffled, "wb"):
                pass
            for name in names:
                tally = {"documents": 0}
                part = staging / f"{phase}-{name}.tokens.bin"
                manifest["tokens"][f"{phase}/{name}"] = write_tokens(texts_of(files[phase, name], recipe, tally),
                                                                     tokenizer, part)
                manifest["documents"][f"{phase}/{name}"] = tally["documents"]
                with open(unshuffled, "ab") as whole, open(part, "rb") as piece:
                    shutil.copyfileobj(piece, whole)
                part.unlink()
            train, counted = split_documents(unshuffled, directory / "train.bin", table / "train.bin", bos,
                                             seed + 1 + number, table_share)
            manifest["tokens"][f"{phase}/train"] = train
            manifest["tokens"][f"{phase}/table"] = counted
            unshuffled.unlink()

        unshuffled = staging / "val.unshuffled.bin"
        with open(unshuffled, "wb"):
            pass
        for name in VALIDATION:
            tally = {"documents": 0}
            part = staging / f"{name}.tokens.bin"
            manifest["tokens"][f"stable/{name}"] = write_tokens(texts_of(files["stable", name], recipe, tally),
                                                                tokenizer, part)
            manifest["documents"][f"stable/{name}"] = tally["documents"]
            with open(unshuffled, "ab") as whole, open(part, "rb") as piece:
                shutil.copyfileobj(piece, whole)
            part.unlink()
        # Both phases read the same validation text, in a random order of its documents.
        split_documents(unshuffled, staging / "stable" / "val.bin", staging / "unused.bin", bos, seed, 0.0)
        (staging / "unused.bin").unlink()
        unshuffled.unlink()
        shutil.rmtree(packed)

        for phase in PHASES:
            directory = staging / phase
            tokenizer.save(str(directory / "tokenizer.json"))
            if phase != "stable":
                shutil.copyfile(staging / "stable" / "val.bin", directory / "val.bin")
            with (directory / "meta.pkl").open("wb") as meta:
                pickle.dump({"vocab_size": tokenizer.get_vocab_size()}, meta)
        tokenizer.save(str(table / "tokenizer.json"))
        manifest["sha256"] = {f"{part}/{name}": file_digest(staging / part / name)
                              for part in (*PHASES, "table") for name in ("tokenizer.json", "train.bin")}
        manifest["sha256"]["stable/val.bin"] = file_digest(staging / "stable" / "val.bin")
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.rename(staging, output)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", help="the new directory, holding stable/, decay/ and table/")
    parser.add_argument("--sources", required=True, help="the file that names the packed files")
    arguments = parser.parse_args()
    sources = json.loads(Path(arguments.sources).read_text(encoding="utf-8"))
    print(json.dumps(prepare_bilingual(arguments.output, sources), indent=2))


if __name__ == "__main__":
    main()
