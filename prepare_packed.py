"""Prepare what training reads from a corpus of language-model's recipe, in prepare.py's tokenizer.

The recipe packs a corpus as 16-bit files of documents, each its tokens between BOS and EOS, one source
after another. Here each document is decoded with the recipe's tokenizer and encoded with prepare.py's, and
the documents are shuffled, as ClimbMix's are, so that training on the result differs from training on
ClimbMix only in the documents it reads.

The sources file is JSON, {"train": {"url": ..., "sha256": ...}, "val": {...}}. It is a file because its URLs
can grant access, and a command's arguments show among a machine's processes.

Usage: python prepare_packed.py OUTPUT --tokenizer TOKENIZER --recipe-tokenizer TOKENIZER --sources FILE
"""

import argparse
import hashlib
import itertools
import json
import os
import pickle
import shutil
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from prepare import BOS, file_digest, write_tokens

EOS = "<|eos|>"
CHUNK_BYTES = 1 << 24
SHUFFLE_SEED = 1337


def read_tokens(stream, digest):
    """The stream's 16-bit tokens in pieces, adding its bytes to the digest."""
    odd = b""
    while chunk := stream.read(CHUNK_BYTES):
        digest.update(chunk)
        chunk = odd + chunk
        whole = len(chunk) - len(chunk) % 2
        odd = chunk[whole:]
        yield np.frombuffer(chunk[:whole], dtype=np.uint16)
    if odd:
        raise ValueError("the file ended inside a token")


def read_documents(pieces, bos, eos):
    """Each document's tokens without its marks. A document's text can spell a mark, which the recipe's
    tokenizer then reads as that mark, so a document ends only where EOS meets the next BOS."""
    unfinished = np.empty(0, dtype=np.uint16)
    for piece in pieces:
        tokens = np.concatenate([unfinished, piece])
        if len(tokens) and tokens[0] != bos:
            raise ValueError("the file does not begin with a document")
        start = 0
        for head in (np.flatnonzero((tokens[:-1] == eos) & (tokens[1:] == bos)) + 1).tolist():
            yield tokens[start + 1 : head - 1]
            start = head
        unfinished = tokens[start:]
    if len(unfinished):
        if unfinished[-1] != eos:
            raise ValueError("the file ended inside a document")
        yield unfinished[1:-1]


def decoded_texts(documents, tokenizer, tally):
    """The documents' texts, counting them in the tally."""
    while batch := list(itertools.islice(documents, 1024)):
        tally["documents"] += len(batch)
        yield from tokenizer.decode_batch([tokens.tolist() for tokens in batch], skip_special_tokens=False)


def shuffle_documents(source, destination, bos, seed):
    """Write the file's documents, each beginning at a BOS, in a random order."""
    tokens = np.memmap(source, dtype=np.uint16, mode="r")
    heads = np.flatnonzero(tokens == bos)
    ends = np.append(heads[1:], len(tokens))
    with open(destination, "wb") as output:
        for index in np.random.default_rng(seed).permutation(len(heads)).tolist():
            output.write(tokens[heads[index] : ends[index]].tobytes())


def prepare_packed(output, tokenizer_path, recipe_tokenizer_path, sources, seed=SHUFFLE_SEED):
    output = Path(output)
    if output.exists():
        raise ValueError("output already exists; select a new directory")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    recipe_tokenizer = Tokenizer.from_file(str(recipe_tokenizer_path))
    marks = recipe_tokenizer.token_to_id(BOS), recipe_tokenizer.token_to_id(EOS)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        counts = {"documents": {}, "tokens": {}}
        for split in ("train", "val"):
            digest, tally = hashlib.sha256(), {"documents": 0}
            unshuffled = staging / f"{split}.unshuffled.bin"
            with urllib.request.urlopen(sources[split]["url"], timeout=120) as stream:
                documents = read_documents(read_tokens(stream, digest), *marks)
                counts["tokens"][split] = write_tokens(decoded_texts(documents, recipe_tokenizer, tally), tokenizer, unshuffled)
            if digest.hexdigest() != sources[split]["sha256"]:
                raise ValueError(f"the {split} file is not the one its source names")
            counts["documents"][split] = tally["documents"]
            shuffle_documents(unshuffled, staging / f"{split}.bin", tokenizer.token_to_id(BOS), seed)
            unshuffled.unlink()
        shutil.copyfile(tokenizer_path, staging / "tokenizer.json")
        with (staging / "meta.pkl").open("wb") as meta:
            pickle.dump({"vocab_size": tokenizer.get_vocab_size()}, meta)
        manifest = {
            "recipe_tokenizer_sha256": file_digest(recipe_tokenizer_path),
            "packed_sha256": {split: sources[split]["sha256"] for split in ("train", "val")},
            "shuffle_seed": seed,
            "vocabulary": tokenizer.get_vocab_size(),
            **counts,
            "sha256": {name: file_digest(staging / name) for name in ("tokenizer.json", "train.bin", "val.bin")},
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.rename(staging, output)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", help="the new directory, which nanoGPT reads as data/<dataset>")
    parser.add_argument("--tokenizer", required=True, help="prepare.py's tokenizer")
    parser.add_argument("--recipe-tokenizer", required=True, help="the tokenizer the packed files are written in")
    parser.add_argument("--sources", required=True, help="the file that names the packed files")
    arguments = parser.parse_args()
    sources = json.loads(Path(arguments.sources).read_text(encoding="utf-8"))
    print(json.dumps(prepare_packed(arguments.output, arguments.tokenizer, arguments.recipe_tokenizer, sources), indent=2))


if __name__ == "__main__":
    main()
