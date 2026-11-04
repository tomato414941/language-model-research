"""Prepare what training reads: FineWeb as nanoGPT's token files, with a tokenizer trained on it.

The files come from a pinned commit of FineWeb's uniform 10B-token sample, which modded-nanogpt reads. The last
file is held out for validation, and training reads the others from the last backwards, so that count_data.py,
which reads them from the first, counts other documents. Each document begins with a BOS token.

Usage: python prepare.py data/fineweb --train-tokens 2000000000
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
import pyarrow.parquet as parquet
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

REPOSITORY = "HuggingFaceFW/fineweb"
REVISION = "9bb295ddab0e05d785b879661af7260fed5140fc"
SHARD_URL = "https://huggingface.co/datasets/{repository}/resolve/{revision}/sample/10BT/{index:03d}_00000.parquet"
LAST_SHARD = 14
# The training files, from the one before the validation file backwards.
TRAIN_SHARDS = range(LAST_SHARD - 1, -1, -1)
# One file holds far more than validation needs.
VALIDATION_TOKENS = 20_000_000
VOCABULARY = 8192
BOS = "<|bos|>"
# GPT-4's split with groups of up to two digits, as autoresearch splits its text.
SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}"""
    r"""| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)
TOKENIZER_CHARACTERS = 1_000_000_000
TOKENIZER_DOCUMENT_CHARACTERS = 10_000


def download_shard(index, directory):
    path = Path(directory) / f"{index:03d}_00000.parquet"
    url = SHARD_URL.format(repository=REPOSITORY, revision=REVISION, index=index)
    with urllib.request.urlopen(url, timeout=120) as source, path.open("wb") as destination:
        shutil.copyfileobj(source, destination)
    return path


def read_texts(path):
    for batch in parquet.ParquetFile(path).iter_batches(columns=["text"], batch_size=1024):
        yield from batch.column("text").to_pylist()


def shard_texts(indices, directory, download):
    """The shards' documents in order, keeping only the shard being read on the disk."""
    for index in indices:
        path = download(index, directory)
        yield from read_texts(path)
        path.unlink()


def train_tokenizer(texts, vocabulary):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(SPLIT_PATTERN), "isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocabulary,
        special_tokens=[BOS],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )

    def sample():
        characters = 0
        for text in texts:
            text = text[:TOKENIZER_DOCUMENT_CHARACTERS]
            characters += len(text)
            yield text
            if characters >= TOKENIZER_CHARACTERS:
                return

    tokenizer.train_from_iterator(sample(), trainer)
    return tokenizer


def write_tokens(texts, tokenizer, path, limit=None):
    """Write the documents' tokens as nanoGPT's 16-bit file, up to the limit; return their count."""
    bos = tokenizer.token_to_id(BOS)
    written = 0
    with open(path, "wb") as output:
        while batch := list(itertools.islice(texts, 1024)):
            for encoding in tokenizer.encode_batch_fast(batch, add_special_tokens=False):
                tokens = np.array([bos, *encoding.ids], dtype=np.uint16)
                if limit is not None:
                    tokens = tokens[: limit - written]
                output.write(tokens.tobytes())
                written += len(tokens)
                if limit is not None and written >= limit:
                    return written
    return written


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while chunk := source.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(output, train_tokens, vocabulary=VOCABULARY, download=download_shard):
    output = Path(output)
    if output.exists():
        raise ValueError("output already exists; select a new directory")
    if vocabulary > 2**16:
        raise ValueError("nanoGPT's files hold 16-bit tokens")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        shards = staging / "shards"
        shards.mkdir()
        tokenizer = train_tokenizer(shard_texts(TRAIN_SHARDS, shards, download), vocabulary)
        tokenizer.save(str(staging / "tokenizer.json"))
        counts = {
            "train": write_tokens(
                shard_texts(TRAIN_SHARDS, shards, download),
                tokenizer,
                staging / "train.bin",
                train_tokens,
            ),
            "val": write_tokens(shard_texts([LAST_SHARD], shards, download), tokenizer, staging / "val.bin",
                                VALIDATION_TOKENS),
        }
        shutil.rmtree(shards)
        if counts["train"] < train_tokens:
            raise ValueError("the shards hold fewer tokens than the training asks for")
        with (staging / "meta.pkl").open("wb") as meta:
            pickle.dump({"vocab_size": tokenizer.get_vocab_size()}, meta)
        manifest = {
            "repository": REPOSITORY,
            "revision": REVISION,
            "validation_shard": LAST_SHARD,
            "vocabulary": tokenizer.get_vocab_size(),
            "tokens": counts,
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
    parser.add_argument("--train-tokens", type=int, required=True)
    parser.add_argument("--vocabulary", type=int, default=VOCABULARY)
    arguments = parser.parse_args()
    print(json.dumps(prepare(arguments.output, arguments.train_tokens, arguments.vocabulary), indent=2))


if __name__ == "__main__":
    main()
