"""Where training rows begin, when they begin at the head of a document.

Finding the heads takes seconds over billions of tokens, so they are kept in a file beside the tokens.
"""

import json
import os
from pathlib import Path

import numpy as np

BOS = "<|bos|>"


def bos_token(tokenizer_path):
    """The number of the token that begins each document, in a tokenizer that prepare.py trained."""
    added = json.loads(Path(tokenizer_path).read_text(encoding="utf-8"))["added_tokens"]
    return next(token["id"] for token in added if token["content"] == BOS)


def document_heads(tokens_path, tokenizer_path):
    """The positions of the tokens that begin a document, from the file beside the tokens if it is there."""
    tokens_path = Path(tokens_path)
    kept = tokens_path.with_suffix(".heads.npy")
    if kept.exists():
        return np.load(kept)
    heads = np.flatnonzero(np.memmap(tokens_path, dtype=np.uint16, mode="r") == bos_token(tokenizer_path))
    unfinished = kept.with_suffix(".unfinished.npy")
    np.save(unfinished, heads)
    os.replace(unfinished, kept)
    return heads


def whole_row_heads(heads, token_count, block_size):
    """The heads from which a whole row and its targets can be read."""
    return heads[heads + block_size + 1 <= token_count]
