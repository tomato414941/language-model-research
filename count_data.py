"""Prepare the text that a COUNT table counts: FineWeb's sample as 16-bit tokens of prepare.py's tokenizer.

The documents come from a pinned commit of FineWeb's uniform 10B-token sample, which training never reads. A
document that holds an evaluation question, found as language-model's v6 rule finds them, is left out, so that
the table cannot hand the model a question's continuation. Each document begins with a BOS token. Finding the
questions needs pyahocorasick.

Usage: python count_data.py data/count-text --tokenizer data/climbmix/tokenizer.json \
           --questions data/questions.jsonl --tokens 2000000000
"""

import argparse
import hashlib
import itertools
import json
import multiprocessing
import os
import re
import shutil
import tempfile
import unicodedata
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as parquet

REPOSITORY = "HuggingFaceFW/fineweb"
REVISION = "9bb295ddab0e05d785b879661af7260fed5140fc"
FILE_URL = "https://huggingface.co/datasets/{repository}/resolve/{revision}/sample/10BT/{index:03d}_00000.parquet"
FILES = 15
BOS = "<|bos|>"
# The rule by which language-model's v6 corpus leaves out the documents that hold evaluation questions.
PREAMBLE = re.compile(r"^The following are questions \(with answers\) about [^\n]*\n\n")
WRAPPER = re.compile(r"^(?:Question:|質問：)\s*|\s*(?:Answer:|回答：)$")
WIDTH, MINIMUM = 64, 24
BATCH = 1024


def normalized(text):
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", text).casefold())


def question_key(context):
    """The middle characters by which a question is found, or None if it is too short to tell from other text."""
    text = normalized(WRAPPER.sub("", PREAMBLE.sub("", context)))
    if len(text) < MINIMUM:
        return None
    width = min(len(text), WIDTH)
    middle = (len(text) - width) // 2
    return text[middle : middle + width]


def question_automaton(questions):
    import ahocorasick

    automaton = ahocorasick.Automaton()
    for key in {question_key(question["context"]) for question in questions} - {None}:
        automaton.add_word(key, key)
    automaton.make_automaton()
    return automaton


# Each worker's own, set by start_worker.
AUTOMATON = TOKENIZER = BOS_ID = None


def start_worker(tokenizer_path, questions_path):
    global AUTOMATON, TOKENIZER, BOS_ID
    from tokenizers import Tokenizer

    TOKENIZER = Tokenizer.from_file(tokenizer_path)
    BOS_ID = TOKENIZER.token_to_id(BOS)
    with open(questions_path, encoding="utf-8") as questions:
        AUTOMATON = question_automaton([json.loads(line) for line in questions])


def holds_question(text):
    return next(AUTOMATON.iter(normalized(text)), None) is not None


def encode_batch(texts):
    """The tokens of the documents that hold no evaluation question, each beginning with BOS, with how many
    documents there were, how many were left out and how many bytes of text the tokens hold."""
    kept = [text for text in texts if not holds_question(text)]
    encodings = TOKENIZER.encode_batch(kept, add_special_tokens=False)
    tokens = [np.array([BOS_ID, *encoding.ids], dtype=np.uint16) for encoding in encodings]
    tokens = np.concatenate(tokens) if tokens else np.zeros(0, dtype=np.uint16)
    return tokens, len(texts), len(texts) - len(kept), sum(len(text.encode("utf-8")) for text in kept)


def download_file(index, directory):
    path = Path(directory) / f"{index:03d}_00000.parquet"
    url = FILE_URL.format(repository=REPOSITORY, revision=REVISION, index=index)
    with urllib.request.urlopen(url, timeout=120) as source, path.open("wb") as destination:
        shutil.copyfileobj(source, destination)
    return path


def file_batches(directory, download):
    """The sample's documents in order and in batches, keeping only the file being read on the disk."""
    for index in range(FILES):
        path = download(index, directory)
        for batch in parquet.ParquetFile(path).iter_batches(columns=["text"], batch_size=BATCH):
            yield batch.column("text").to_pylist()
        path.unlink()


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while chunk := source.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(output, tokenizer_path, questions_path, tokens, download=download_file, workers=4):
    output = Path(output)
    if output.exists():
        raise ValueError("output already exists; select a new directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    # Each worker tokenizes on one thread, so that the threads are not shared across the fork.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        files = staging / "files"
        files.mkdir()
        written = documents = left_out = text_bytes = 0
        context = multiprocessing.get_context("fork")
        arguments = (str(tokenizer_path), str(questions_path))
        with context.Pool(workers, start_worker, arguments) as pool, open(staging / "train.bin", "wb") as out:
            batches = file_batches(files, download)
            for batch_tokens, batch_documents, batch_left_out, batch_bytes in pool.imap(encode_batch, batches):
                kept = batch_tokens[: tokens - written]
                out.write(kept.tobytes())
                written += len(kept)
                documents += batch_documents
                left_out += batch_left_out
                text_bytes += batch_bytes * len(kept) // max(1, len(batch_tokens))
                if written >= tokens:
                    break
        shutil.rmtree(files)
        if written < tokens:
            raise ValueError("the sample holds fewer tokens than asked for")
        manifest = {
            "repository": REPOSITORY,
            "revision": REVISION,
            "documents_read": documents,
            "documents_with_a_question": left_out,
            "tokens": written,
            "bytes_per_token": round(text_bytes / written, 3),
            "sha256": {"tokenizer.json": file_digest(tokenizer_path), "train.bin": file_digest(staging / "train.bin")},
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.rename(staging, output)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", help="the new directory for the tokens")
    parser.add_argument("--tokenizer", required=True, help="prepare.py's tokenizer.json")
    parser.add_argument("--questions", required=True, help="the questions that harness.py wrote out")
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args()
    manifest = prepare(arguments.output, arguments.tokenizer, arguments.questions, arguments.tokens,
                       workers=arguments.workers)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
