"""Write a dataset that train.py reads in prepare.py's kind of tokenizer as bytes for H-Net: the same documents in the
same order, each token replaced by its bytes and BOS by the byte model's BOS (256), so that H-Net reads exactly what
the BPE model reads.

Writes OUTPUT/train.bin, val.bin, meta.pkl, rows.json, whose block_size holds as many bytes as 1,024 of the source's
tokens hold on average, so that a row of either model holds the same text, and tokenizer.json: a tokenizer, in the
format that prepare.py's is saved in, whose tokens are the 256 byte values and BOS, so that what reads a dataset's
tokenizer (rows.py's BOS, frame.py's tokenizer for olmes) reads the bytes as they are.

Usage: python h-net/bytes_from_tokens.py SOURCE OUTPUT, both directories as train.py reads them (data/<dataset>)
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "h-net")]

from hnet_model import BOS as BYTE_BOS  # noqa: E402
from prepare import BOS  # noqa: E402

ROW_TOKENS = 1024
CHUNK_TOKENS = 1 << 24


def byte_alphabet():
    """The byte that each character of a byte-level BPE token stands for, as GPT-2 and tokenizers' ByteLevel map
    them: a printable byte is its own character, and the others take the characters from 256 on, in byte order."""
    printable = {*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)}
    table, others = {}, 0
    for byte in range(256):
        if byte in printable:
            table[chr(byte)] = byte
        else:
            table[chr(256 + others)] = byte
            others += 1
    return table


def byte_tokenizer():
    """The 256 byte values as tokens numbered by their value, through the byte-level characters of byte_alphabet and
    no merges, and BOS as the 257th."""
    from tokenizers import AddedToken, decoders, models, pre_tokenizers

    tokenizer = Tokenizer(models.BPE(vocab=dict(byte_alphabet()), merges=[]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens([AddedToken(BOS, special=True)])
    assert tokenizer.token_to_id(BOS) == BYTE_BOS
    return tokenizer


class TokenBytes:
    """Each token's bytes, as 16-bit values in one array, with BOS as the byte model's BOS."""

    def __init__(self, tokenizer):
        alphabet = byte_alphabet()
        pieces = [None] * tokenizer.get_vocab_size(with_added_tokens=True)
        for token, index in tokenizer.get_vocab(with_added_tokens=True).items():
            pieces[index] = [BYTE_BOS] if token == BOS else [alphabet[character] for character in token]
        self.lengths = np.array([len(piece) for piece in pieces], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.lengths)[:-1]])
        self.flat = np.array([value for piece in pieces for value in piece], dtype=np.uint16)

    def of(self, tokens):
        """The bytes of a stretch of tokens, one after another."""
        tokens = np.asarray(tokens, dtype=np.int64)
        lengths = self.lengths[tokens]
        ends = np.cumsum(lengths)
        return self.flat[np.repeat(self.offsets[tokens] - (ends - lengths), lengths) + np.arange(int(ends[-1]))]


def convert(source, output, table):
    """Write a token file's bytes; return the counts of tokens and bytes, and of BOS, which both hold once."""
    tokens = np.memmap(source, dtype=np.uint16, mode="r")
    counts = {"tokens": len(tokens), "bytes": 0, "documents": 0}
    with open(output, "wb") as written:
        for start in range(0, len(tokens), CHUNK_TOKENS):
            values = table.of(tokens[start : start + CHUNK_TOKENS])
            written.write(values.tobytes())
            counts["bytes"] += len(values)
            counts["documents"] += int((values == BYTE_BOS).sum())
    return counts


def bytes_from_tokens(source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True)
    table = TokenBytes(Tokenizer.from_file(str(source / "tokenizer.json")))
    counts = {split: convert(source / f"{split}.bin", output / f"{split}.bin", table) for split in ("train", "val")}
    with (output / "meta.pkl").open("wb") as meta:
        pickle.dump({"vocab_size": BYTE_BOS + 1}, meta)
    byte_tokenizer().save(str(output / "tokenizer.json"))
    train = counts["train"]
    bytes_per_token = (train["bytes"] - train["documents"]) / (train["tokens"] - train["documents"])
    # A multiple of 256, which the outer layers' kernels split rows into.
    block_size = int(round(ROW_TOKENS * bytes_per_token / 256)) * 256
    (output / "rows.json").write_text(json.dumps({"block_size": block_size, "bytes_per_token": bytes_per_token,
                                                  "source": source.name, "counts": counts}, indent=1) + "\n")
    return block_size, counts


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    block_size, counts = bytes_from_tokens(*sys.argv[1:])
    print(json.dumps({"block_size": block_size, **counts}))


if __name__ == "__main__":
    main()
