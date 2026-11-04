"""Scores an H-Net model on what rented.sh scores a BPE model on: the validation bits per byte over the same rows of
the BPE dataset's val.bin that scoring.py reads (its first 2**20 tokens, in rows of 1,024 tokens, each after the token
before it), and harness.py's English and Japanese benchmarks in full.

Each row's bytes are read after the bytes of the token before it, and a row longer than H-Net's context in pieces,
each after as much of the row before it as fits. The bits are divided by the bytes that scoring.py counts for the row,
so that the two models' bits per byte share a denominator.

With --validation-only it reads the validation rows alone, and needs no lm-eval.

Usage: python h-net/score_hnet.py OUTPUT MODEL_DIRECTORY --tokens DATASET_DIRECTORY [--limit N] [--validation-only]
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

sys.path[:0] = [".", "h-net"]
from bytes_from_tokens import TokenBytes  # noqa: E402
from hnet_model import byte_language_model, load  # noqa: E402
from loading import loglikelihoods  # noqa: E402

BPE_CONTEXT = 1024
VALIDATION_TOKENS = 2**20


def validation_bits(model, dataset):
    tokenizer = Tokenizer.from_file(str(Path(dataset) / "tokenizer.json"))
    table = TokenBytes(tokenizer)
    data = np.fromfile(Path(dataset) / "val.bin", dtype=np.uint16, count=VALIDATION_TOKENS)
    rows = [data[start : start + BPE_CONTEXT + 1] for start in range(0, len(data) - 1, BPE_CONTEXT)]
    requests, size = [], 0
    for row in (row for row in rows if len(row) > 1):
        before, text = table.of(row[:1]).tolist(), table.of(row[1:]).tolist()
        requests += [(before + text[:start], text[start : start + model.context])
                     for start in range(0, len(text), model.context)]
        size += len(tokenizer.decode(row[1:].tolist()).encode("utf-8"))
    bits = -sum(result[0] for result in loglikelihoods(model, requests)) / math.log(2)
    return {"bits_per_byte": bits / size, "bytes": size, "pieces": len(requests)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output")
    parser.add_argument("model", help="train.py's output directory")
    parser.add_argument("--tokens", required=True, help="the BPE dataset whose val.bin scoring.py reads")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-tokens", type=int, default=65536)
    parser.add_argument("--limit", type=int, help="ask only this many questions of each task, to try it")
    parser.add_argument("--validation-only", action="store_true", help="read the validation rows alone")
    arguments = parser.parse_args()
    model = byte_language_model(load(arguments.model, arguments.device), arguments.device, arguments.batch_tokens)
    result = {"validation": validation_bits(model, arguments.tokens)}
    summary = {"validation_bits_per_byte": round(result["validation"]["bits_per_byte"], 4)}
    if arguments.validation_only:
        Path(arguments.output).write_text(json.dumps(result) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=1))
        return
    from harness import BENCHMARKS, score_in_full

    for language in ("en", "ja"):
        benchmarks = {name: row for name, row in BENCHMARKS.items() if row[0] == language}
        result[language] = score_in_full(model, benchmarks, arguments.limit)
    Path(arguments.output).write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
    for language in ("en", "ja"):
        for name, row in result[language]["benchmarks"].items():
            summary[name] = {"accuracy": round(row["accuracy"], 4), "gold_bpb": round(row["gold_bits_per_byte"], 4)}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
