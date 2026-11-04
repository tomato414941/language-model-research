"""COUNT tables (arXiv 2610.20456): for each history of up to five tokens seen often enough in a text that training
does not read, the eight tokens that most often follow it, scored by how often they do.

The model looks up the table of each history length at every position and adds the output table's rows of the
candidates, weighted by their scores, to its input. The table is counted once, before training, and does not
change while the model learns; the rows it points to are the model's own and do.

A history's key is the same arithmetic in NumPy, where the table is built, and in PyTorch, where the model looks
it up: products wrap around in 64 bits in both, and shifts are masked to act as unsigned ones.

Usage: python count_table.py data/count-jaen --text data/jaen/table --tokenizer data/jaen/table/tokenizer.json
"""

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

LENGTHS = (1, 2, 3, 4, 5)  # history lengths that have their own tables; the empty history has one row
ORDERS = len(LENGTHS) + 1
K = 8
VOCABULARY = 8192
NEXT_BITS = 13  # a key leaves its low bits to the next token while the table is counted
KEY_MASK = -(1 << NEXT_BITS)
EMPTY = 255  # the score of a slot with no candidate, which weighs nothing
STEPS_PER_HALVING = 16  # scores are stored as -log2(score) in sixteenths, in one byte


def signed(value):
    return value - (1 << 64) if value >= 1 << 63 else value


PRIME = signed(0x9E3779B97F4A7C15)
MIX = signed(0xC2B2AE3D27D4EB4F)
LOW_35 = (1 << 35) - 1


def mixed(hashed):
    """Spreads the bits of a 64-bit polynomial hash, for NumPy and PyTorch arrays alike."""
    hashed = hashed * MIX
    hashed = hashed ^ ((hashed >> 29) & LOW_35)
    hashed = hashed * PRIME
    return hashed & KEY_MASK


def history_keys(values, length):
    """The keys of the histories of `length` tokens that end at positions length-1 to n-1 of the last axis.

    `values` holds the tokens plus one, as 64-bit signed integers, in NumPy or PyTorch.
    """
    n = values.shape[-1]
    hashed = values[..., length - 1 :] * 0
    for back in range(length):
        hashed = hashed * PRIME + values[..., length - 1 - back : n - back]
    return mixed(hashed)


def quantized(scores):
    """Scores in (0, 1] as sixteenths of halvings, below the value kept for empty slots."""
    steps = np.rint(-np.log2(scores) * STEPS_PER_HALVING)
    return np.clip(steps, 0, EMPTY - 1).astype(np.uint8)


def score_levels():
    """The score that each stored byte stands for."""
    levels = 2.0 ** (-np.arange(256) / STEPS_PER_HALVING)
    levels[EMPTY] = 0.0
    return levels.astype(np.float32)


def balanced_parts(counts, parts):
    """Assigns each token to one of `parts` groups, so that the groups' summed counts are close."""
    part_of = np.zeros(len(counts), dtype=np.uint8)
    loads = np.zeros(parts, dtype=np.int64)
    for token in np.argsort(-counts, kind="stable"):
        part = int(np.argmin(loads))
        part_of[token] = part
        loads[part] += counts[token]
    return part_of


def top_continuations(pairs, counts, floor, k):
    """For each history seen at least `floor` times, its k most frequent next tokens and their scores."""
    history = pairs & KEY_MASK
    starts = np.flatnonzero(np.concatenate(([True], history[1:] != history[:-1]))) if len(pairs) else pairs
    sizes = np.diff(np.concatenate((starts, [len(pairs)])))
    occurrences = np.add.reduceat(counts, starts) if len(starts) else np.zeros(0, dtype=np.int64)
    kept = occurrences >= floor
    if not kept.any():
        return np.zeros(0, dtype=np.int64), np.zeros((0, k), dtype=np.uint16), np.zeros((0, k), dtype=np.uint8)
    wanted = np.repeat(kept, sizes)
    pairs, counts = pairs[wanted], counts[wanted]
    sizes, occurrences = sizes[kept], occurrences[kept]
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1])).astype(np.int64)
    history = pairs & KEY_MASK
    order = np.lexsort((pairs & ~KEY_MASK, -counts, history))
    ranks = np.arange(len(pairs)) - np.repeat(starts, sizes)
    chosen = order[ranks < k]
    row = np.repeat(np.arange(len(starts)), np.minimum(sizes, k))
    slot = ranks[ranks < k]
    tokens = np.zeros((len(starts), k), dtype=np.uint16)
    scores = np.full((len(starts), k), EMPTY, dtype=np.uint8)
    tokens[row, slot] = (pairs[chosen] & ~KEY_MASK).astype(np.uint16)
    scores[row, slot] = quantized(counts[chosen] / np.repeat(occurrences, np.minimum(sizes, k)))
    return history[starts], tokens, scores


def count_pairs(tokens, length, part_of, part, bos, chunk):
    """Each counted position's history key with its next token in the low bits, for one group of the histories'
    last tokens. A history may begin with BOS but holds no other."""
    pieces = []
    for start in range(0, len(tokens) - length, chunk):
        window = np.asarray(tokens[start : min(len(tokens), start + chunk + length)])
        positions = len(window) - length
        seen = np.cumsum(window == bos)
        valid = seen[length - 1 : length - 1 + positions] == seen[:positions]
        chosen = np.flatnonzero(valid & (part_of[window[length - 1 : length - 1 + positions]] == part))
        values = window.astype(np.int64) + 1
        hashed = chosen.astype(np.int64) * 0
        for back in range(length):
            hashed = hashed * PRIME + values[chosen + (length - 1 - back)]
        pieces.append(mixed(hashed) | window[chosen + length].astype(np.int64))
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.int64)


def sorted_counts(pairs):
    """The distinct values of a sorted array and how often each occurs."""
    if not len(pairs):
        return pairs, np.zeros(0, dtype=np.int64)
    starts = np.concatenate(([0], np.flatnonzero(pairs[1:] != pairs[:-1]) + 1))
    return pairs[starts], np.diff(np.concatenate((starts, [len(pairs)])))


def build(text, output, tokenizer, floor=4, k=K, parts=16, chunk=50_000_000, vocabulary=VOCABULARY):
    """Counts the tokens in `text`/train.bin into a table in the new directory `output`."""
    from tokenizers import Tokenizer

    text, output = Path(text), Path(output)
    if output.exists():
        raise ValueError("output already exists; select a new directory")
    bos = Tokenizer.from_file(str(tokenizer)).token_to_id("<|bos|>")
    tokens = np.memmap(text / "train.bin", dtype=np.uint16, mode="r")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        counts = np.zeros(vocabulary, dtype=np.int64)
        for start in range(0, len(tokens), chunk):
            counts += np.bincount(np.asarray(tokens[start : start + chunk]), minlength=vocabulary)
        best = np.argsort(-counts, kind="stable")[:k]
        np.save(staging / "tokens0.npy", best.astype(np.uint16))
        np.save(staging / "scores0.npy", quantized(np.maximum(counts[best], 1) / counts.sum()))
        part_of = balanced_parts(counts, parts)
        rows = {}
        for length in LENGTHS:
            began = time.time()
            keys, candidates, scores = [], [], []
            for part in range(parts):
                pairs = count_pairs(tokens, length, part_of, part, bos, chunk)
                pairs.sort()
                pairs, pair_counts = sorted_counts(pairs)
                part_keys, part_candidates, part_scores = top_continuations(pairs, pair_counts, floor, k)
                del pairs, pair_counts
                keys.append(part_keys)
                candidates.append(part_candidates)
                scores.append(part_scores)
            keys = np.concatenate(keys)
            order = np.argsort(keys, kind="stable")
            np.save(staging / f"keys{length}.npy", keys[order])
            np.save(staging / f"tokens{length}.npy", np.concatenate(candidates)[order])
            np.save(staging / f"scores{length}.npy", np.concatenate(scores)[order])
            rows[length] = int(len(keys))
            print(json.dumps({"history_tokens": length, "rows": rows[length], "seconds": round(time.time() - began, 1)}),
                  flush=True)
            del keys, candidates, scores, order
        manifest = {
            "text": json.loads((text / "manifest.json").read_text()) if (text / "manifest.json").exists() else None,
            "tokens": int(len(tokens)),
            "floor": floor,
            "k": k,
            "bos": int(bos),
            "rows": rows,
            "bytes": sum(f.stat().st_size for f in staging.iterdir()),
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.rename(staging, output)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def check_device_keys(device):
    """Makes sure that the device computes the keys that the CPU does, since the table's keys depend on 64-bit
    products wrapping around."""
    import torch

    values = torch.randint(1, VOCABULARY + 1, (2, 64), generator=torch.Generator().manual_seed(0))
    for length in LENGTHS:
        if not torch.equal(history_keys(values.to(device), length).cpu(), history_keys(values, length)):
            raise RuntimeError(f"{device} computes other COUNT keys than the CPU does")


class CountTable:
    """A table on a device, as the model looks it up."""

    def __init__(self, directory, device="cpu"):
        import torch

        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        self.bos = manifest["bos"]
        load = lambda name: torch.from_numpy(np.load(directory / f"{name}.npy"))
        # Token numbers fit in 16 signed bits, which PyTorch can index with.
        load_tokens = lambda name: torch.from_numpy(np.load(directory / f"{name}.npy").view(np.int16))
        self.tokens = [load_tokens("tokens0").to(device)]
        self.scores = [load("scores0").to(device)]
        self.keys = [None]
        for length in LENGTHS:
            self.keys.append(load(f"keys{length}").to(device))
            self.tokens.append(load_tokens(f"tokens{length}").to(device))
            self.scores.append(load(f"scores{length}").to(device))
        self.levels = torch.from_numpy(score_levels()).to(device)
        check_device_keys(device)

    def candidates(self, idx):
        """For each position of the rows `idx`, the candidates of its history of each length that the table holds,
        and their weights, which sum over the lengths to an average."""
        import torch

        b, t = idx.shape
        k = self.tokens[0].numel()
        values = idx.long() + 1
        seen = torch.cumsum(idx == self.bos, dim=1)
        tokens = [self.tokens[0].expand(b, t, k)]
        weights = [self.levels[self.scores[0].long()].expand(b, t, k)]
        for length in LENGTHS:
            if length > t or not len(self.keys[length]):
                tokens.append(torch.zeros(b, t, k, dtype=torch.int16, device=idx.device))
                weights.append(torch.zeros(b, t, k, device=idx.device))
                continue
            keys = history_keys(values, length)
            valid = seen[:, length - 1 :] == seen[:, : t - length + 1]
            table = self.keys[length]
            place = torch.searchsorted(table, keys).clamp(max=len(table) - 1)
            found = (table[place] == keys) & valid
            weight = self.levels[self.scores[length][place].long()] * found[..., None]
            gap = (b, length - 1, k)
            tokens.append(torch.cat((torch.zeros(gap, dtype=torch.int16, device=idx.device), self.tokens[length][place]), 1))
            weights.append(torch.cat((torch.zeros(gap, device=idx.device), weight), 1))
        return torch.cat(tokens, -1).long(), torch.cat(weights, -1) / ORDERS


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", help="the new directory for the table")
    parser.add_argument("--text", required=True, help="count_data.py's directory")
    parser.add_argument("--tokenizer", required=True, help="the tokenizer.json that count_data.py tokenized with")
    parser.add_argument("--floor", type=int, default=4, help="how often a history must occur to have a row")
    parser.add_argument("--parts", type=int, default=16)
    arguments = parser.parse_args()
    manifest = build(arguments.text, arguments.output, arguments.tokenizer, arguments.floor, parts=arguments.parts)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
