"""Prepare the other allowed data beside FineWeb's: FineMath's two subsets and Common Pile's educational code, each
as nanoGPT's token file in a directory of its own, in the tokenizer that prepare.py trained on FineWeb.

Each set is read at the revision that frame.py pins, from its first files in the order of their names, or from the
first documents of each file, read as they arrive: Common Pile's code files are ordered by language (the first twenty
hold only Markdown), so its languages come in as the set has them only when every file gives a share. Nothing is chosen
by its content. Each document begins with a BOS token. How much of each a run reads is up to the run.

Usage: python prepare_extra.py NAME OUTPUT --tokenizer data/fineweb/tokenizer.json [--files 2] [--documents-per-file N]
"""

import argparse
import gzip
import itertools
import json
import os
import pickle
import shutil
import tempfile
import urllib.request
from pathlib import Path

from tokenizers import Tokenizer

from frame import ALLOWED_DATA
from prepare import file_digest, read_texts, write_tokens

URL = "https://huggingface.co/datasets/{repository}/resolve/{revision}/{path}"


def data_files(name):
    """The set's data files at its revision, in the order of their names."""
    from huggingface_hub import HfApi

    repository, revision, folder = ALLOWED_DATA[name]
    tree = HfApi().list_repo_tree(repository, repo_type="dataset", revision=revision, path_in_repo=folder or None)
    return sorted(item.path for item in tree if item.path.endswith((".parquet", ".json.gz")))


def download(name, path, directory):
    repository, revision, _ = ALLOWED_DATA[name]
    target = Path(directory) / Path(path).name
    with urllib.request.urlopen(URL.format(repository=repository, revision=revision, path=path), timeout=120) as source:
        with target.open("wb") as destination:
            shutil.copyfileobj(source, destination)
    return target


def open_remote(name, path):
    repository, revision, _ = ALLOWED_DATA[name]
    return urllib.request.urlopen(URL.format(repository=repository, revision=revision, path=path), timeout=120)


def first_texts(name, path, documents, open_file=open_remote):
    """The first documents of one file of gzipped JSON, read as it arrives, so that only that part is fetched."""
    with open_file(name, path) as source, gzip.open(source, "rt", encoding="utf-8") as lines:
        for line in itertools.islice(lines, documents):
            yield json.loads(line)["text"]


def file_texts(path):
    """The documents of one data file: a Parquet file's text column, or the text of each line of gzipped JSON."""
    path = Path(path)
    if path.name.endswith(".json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as lines:
            for line in lines:
                yield json.loads(line)["text"]
    else:
        yield from read_texts(path)


def texts(name, paths, directory, fetch):
    """The documents of the files in order, keeping only the file being read on the disk."""
    for path in paths:
        local = fetch(name, path, directory)
        yield from file_texts(local)
        local.unlink()


def prepare(name, output, tokenizer_path, files=None, fetch=download, list_files=data_files, documents_per_file=None,
            open_file=open_remote):
    """Writes the set's tokens to output: the first `files` files (every file when None), whole, or only the first
    documents_per_file documents of each."""
    output = Path(output)
    if output.exists():
        raise ValueError("output already exists; select a new directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        paths = list_files(name)[:files]
        if files is not None and len(paths) < files:
            raise ValueError(f"{name} has only {len(paths)} data files")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        if documents_per_file:
            if not all(path.endswith(".json.gz") for path in paths):
                raise ValueError("only files of gzipped JSON can be read in part")
            documents = itertools.chain.from_iterable(first_texts(name, path, documents_per_file, open_file)
                                                      for path in paths)
            count = write_tokens(documents, tokenizer, staging / "train.bin")
        else:
            downloads = staging / "downloads"
            downloads.mkdir()
            count = write_tokens(texts(name, paths, downloads, fetch), tokenizer, staging / "train.bin")
            shutil.rmtree(downloads)
        shutil.copyfile(tokenizer_path, staging / "tokenizer.json")
        with (staging / "meta.pkl").open("wb") as meta:
            pickle.dump({"vocab_size": tokenizer.get_vocab_size()}, meta)
        repository, revision, folder = ALLOWED_DATA[name]
        manifest = {
            "repository": repository,
            "revision": revision,
            "folder": folder,
            "files": paths,
            "documents_per_file": documents_per_file,
            "tokens": count,
            "sha256": {file: file_digest(staging / file) for file in ("tokenizer.json", "train.bin")},
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.rename(staging, output)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", choices=[name for name in ALLOWED_DATA if name != "fineweb"])
    parser.add_argument("output", help="the new directory, which nanoGPT reads as data/<dataset>")
    parser.add_argument("--tokenizer", required=True, help="the tokenizer.json that prepare.py trained")
    parser.add_argument("--files", type=int, help="how many of the first data files to read; every file when left out")
    parser.add_argument("--documents-per-file", type=int, help="read only this many of each file's first documents")
    arguments = parser.parse_args()
    print(json.dumps(prepare(arguments.name, arguments.output, arguments.tokenizer, arguments.files,
                             documents_per_file=arguments.documents_per_file), indent=2))


if __name__ == "__main__":
    main()
