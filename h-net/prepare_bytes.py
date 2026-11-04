"""Prepare the H-Net comparison's data from language-model's v6 corpora in Japanese and English: the same documents
as bytes for H-Net, and in a tokenizer of prepare.py's kind for the BPE model.

For each phase of the v6 recipe, stable and decay, the documents of train_ja.bin and train_en.bin are taken in one
random order, decoded with the recipe's tokenizer, until the texts reach the phase's bytes. Both files hold the same
number of tokens, so the mix keeps the recipe's half of each language. The BPE model's tokenizer is trained as
prepare.py trains its own, with 8,192 tokens, on the stable phase's texts: ClimbMix, whose license forbids commercial
use, is not read. The validation documents are the stable corpus's, kept by language for scoring.

Writes, under the data directory, hnet-bpe-stable, hnet-bpe-decay, hnet-bytes-stable and hnet-bytes-decay, each with
train.bin, val.bin, meta.pkl and tokenizer.json as train.py and rows.py read them, and hnet-validation.jsonl. A byte
row holds as many bytes as 1,024 of the BPE model's tokens hold on average, so that a row of either model reads the
same text; hnet-bytes-stable/rows.json gives that size.

The sources file is JSON: {"tokenizer": {"url", "sha256"}, "stable": {"train_ja", "train_en", "validation_ja",
"validation_en"}, "decay": {"train_ja", "train_en"}}, each part a {"url", "sha256"}. It is a file because its URLs grant
access, and a command's arguments show among a machine's processes.

Usage: python h-net/prepare_bytes.py --sources FILE [--data DIRECTORY] [--phase-bytes N]
"""

import argparse
import hashlib
import itertools
import json
import pickle
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "h-net")]

from hnet_model import BOS as BYTE_BOS  # noqa: E402
from prepare import BOS, VOCABULARY, file_digest, train_tokenizer, write_tokens  # noqa: E402
from prepare_packed import EOS, read_documents  # noqa: E402

SEED = 1337
PHASE_BYTES = 2_500_000_000  # about twice what a run reads in each phase
ROW_TOKENS = 1024
CHUNK_BYTES = 1 << 24


def download(source, path):
    """Fetch a file, checking it against its SHA-256. The URL is never printed."""
    digest = hashlib.sha256()
    with urllib.request.urlopen(source["url"], timeout=120) as stream, open(path, "wb") as output:
        while chunk := stream.read(CHUNK_BYTES):
            digest.update(chunk)
            output.write(chunk)
    if digest.hexdigest() != source["sha256"]:
        raise ValueError(f"{path.name} is not the file its source names")
    return path


def documents(path, marks):
    """The documents of a recipe's file, each as its tokens without the marks."""
    return list(read_documents([np.fromfile(path, dtype=np.uint16)], *marks))


def mixed_texts(files, recipe, marks, budget, seed):
    """The texts of the files' documents in one random order, until they hold `budget` bytes, with each text's
    language."""
    pool = [(language, tokens) for language, path in files.items() for tokens in documents(path, marks)]
    order = np.random.default_rng(seed).permutation(len(pool)).tolist()
    texts, size = [], 0
    for start in range(0, len(order), 1024):
        batch = [pool[index] for index in order[start : start + 1024]]
        decoded = recipe.decode_batch([tokens.tolist() for _, tokens in batch], skip_special_tokens=False)
        for (language, _), text in zip(batch, decoded):
            texts.append((language, text))
            size += len(text.encode("utf-8"))
        if size >= budget:
            break
    return texts


def write_bytes(texts, path):
    """Write each text as BOS and its UTF-8 bytes, in 16 bits as nanoGPT reads tokens; return the count."""
    written = 0
    with open(path, "wb") as output:
        for text in texts:
            row = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.uint16)
            output.write(np.concatenate([[BYTE_BOS], row]).astype(np.uint16).tobytes())
            written += 1 + len(row)
    return written


def byte_dataset(directory, train, val):
    directory.mkdir(parents=True)
    counts = {"train": write_bytes(train, directory / "train.bin"), "val": write_bytes(val, directory / "val.bin")}
    with (directory / "meta.pkl").open("wb") as meta:
        pickle.dump({"vocab_size": BYTE_BOS + 1}, meta)
    # rows.py finds the BOS of each document by this file's added tokens.
    added = {"added_tokens": [{"id": BYTE_BOS, "content": BOS}]}
    (directory / "tokenizer.json").write_text(json.dumps(added) + "\n", encoding="utf-8")
    return counts


def token_dataset(directory, train, val, tokenizer, tokenizer_path):
    directory.mkdir(parents=True)
    counts = {"train": write_tokens(iter(train), tokenizer, directory / "train.bin"),
              "val": write_tokens(iter(val), tokenizer, directory / "val.bin")}
    shutil.copyfile(tokenizer_path, directory / "tokenizer.json")
    with (directory / "meta.pkl").open("wb") as meta:
        pickle.dump({"vocab_size": tokenizer.get_vocab_size()}, meta)
    return counts


def prepare(sources, data, phase_bytes=PHASE_BYTES, seed=SEED):
    data = Path(data)
    staging = data / "hnet-sources"
    staging.mkdir(parents=True, exist_ok=True)
    recipe = Tokenizer.from_file(str(download(sources["tokenizer"], staging / "recipe-tokenizer.json")))
    marks = recipe.token_to_id(BOS), recipe.token_to_id(EOS)

    validation = []
    for language in ("ja", "en"):
        path = download(sources["stable"][f"validation_{language}"], staging / f"validation_{language}.bin")
        tokens = documents(path, marks)
        validation += [(language, text) for text in recipe.decode_batch([t.tolist() for t in tokens],
                                                                         skip_special_tokens=False)]
    with (data / "hnet-validation.jsonl").open("w", encoding="utf-8") as output:
        for language, text in validation:
            output.write(json.dumps({"language": language, "text": text}, ensure_ascii=False) + "\n")
    val_texts = [text for _, text in validation]

    phases = {}
    for phase in ("stable", "decay"):
        files = {language: download(sources[phase][f"train_{language}"], staging / f"{phase}_{language}.bin")
                 for language in ("ja", "en")}
        phases[phase] = mixed_texts(files, recipe, marks, phase_bytes, seed)
        for path in files.values():
            path.unlink()

    tokenizer_path = staging / "tokenizer.json"
    tokenizer = train_tokenizer((text for _, text in phases["stable"]), VOCABULARY)
    tokenizer.save(str(tokenizer_path))
    manifest = {"seed": seed, "phase_bytes": phase_bytes, "sources_sha256": {
        "tokenizer": sources["tokenizer"]["sha256"],
        **{f"{phase}_{name}": part["sha256"] for phase in ("stable", "decay") for name, part in sources[phase].items()},
    }, "datasets": {}}
    for phase, texts in phases.items():
        train = [text for _, text in texts]
        manifest["datasets"][phase] = {
            "documents": {language: sum(1 for lang, _ in texts if lang == language) for language in ("ja", "en")},
            "text_bytes": {language: sum(len(t.encode("utf-8")) for lang, t in texts if lang == language)
                           for language in ("ja", "en")},
            "bpe_tokens": token_dataset(data / f"hnet-bpe-{phase}", train, val_texts, tokenizer, tokenizer_path),
            "bytes": byte_dataset(data / f"hnet-bytes-{phase}", train, val_texts),
        }
    stable = manifest["datasets"]["stable"]
    # Each document's BOS takes a place in both files, so it is left out of the bytes each token holds.
    count = sum(stable["documents"].values())
    bytes_per_token = (stable["bytes"]["train"] - count) / (stable["bpe_tokens"]["train"] - count)
    # A multiple of 256, which the outer layers' kernels split rows into.
    block_size = int(round(ROW_TOKENS * bytes_per_token / 256)) * 256
    manifest["bytes_per_token"] = bytes_per_token
    manifest["block_size"] = block_size
    (data / "hnet-bytes-stable" / "rows.json").write_text(json.dumps({"block_size": block_size}) + "\n")
    manifest["tokenizer_sha256"] = file_digest(tokenizer_path)
    (data / "hnet-manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    shutil.rmtree(staging)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sources", required=True, help="the file that names the corpora's files")
    parser.add_argument("--data", default="data")
    parser.add_argument("--phase-bytes", type=int, default=PHASE_BYTES)
    arguments = parser.parse_args()
    sources = json.loads(Path(arguments.sources).read_text(encoding="utf-8"))
    manifest = prepare(sources, arguments.data, arguments.phase_bytes)
    print(json.dumps({key: value for key, value in manifest.items() if key != "sources_sha256"}, indent=1))


if __name__ == "__main__":
    main()
