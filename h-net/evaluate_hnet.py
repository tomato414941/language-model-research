"""Evaluates an H-Net that train.py left, as frame.py evaluates a run of model.py: the validation BPB over the rows of
FineWeb's held-out file that scoring.py reads, then the three OLMES suites through frame.py's own runner, with each
text's UTF-8 bytes as its tokens. It runs where olmes is installed (olmes.txt), with mamba-ssm beside it.

Usage: python h-net/evaluate_hnet.py OUT_DIR REQUESTS_DIR [DEVICE], from the repository's root
"""

import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path[:0] = [".", "h-net"]
import frame  # noqa: E402
from hnet_model import byte_language_model, load  # noqa: E402

# The longest row that olmes's wrapper passes to the model, in bytes. A row must hold the longest answer of the three
# suites, 5,314 bytes; frame.py's 4,096 tokens hold about 15,600 bytes of FineWeb's text. OlmesModel reads a row
# longer than the model's context in windows that reach back only as far as the task's longest answer needs.
OLMES_LENGTH = 4 * frame.OLMES_LENGTH
VALIDATION_TOKENS = 2**20
BPE_CONTEXT = 1024
VALIDATION_BATCH_BYTES = 65536


def validation_bits(model, dataset):
    """scoring.py's validation BPB for a model of bytes: the same rows of the BPE dataset's val.bin (its first 2**20
    tokens, in rows of 1,024 tokens, each after the token before it). Each row's bytes are read after the bytes of the
    token before it, and a row longer than the model's context in pieces, each after as much of the row before it as
    fits. The bits are divided by the bytes that scoring.py counts for the row, so that a BPE model's bits per byte
    and this share a denominator."""
    from tokenizers import Tokenizer

    from bytes_from_tokens import TokenBytes
    from loading import loglikelihoods

    tokenizer = Tokenizer.from_file(os.path.join(dataset, "tokenizer.json"))
    table = TokenBytes(tokenizer)
    data = np.fromfile(os.path.join(dataset, "val.bin"), dtype=np.uint16, count=VALIDATION_TOKENS)
    rows = [data[start : start + BPE_CONTEXT + 1] for start in range(0, len(data) - 1, BPE_CONTEXT)]
    requests, size = [], 0
    for row in (row for row in rows if len(row) > 1):
        before, text = table.of(row[:1]).tolist(), table.of(row[1:]).tolist()
        requests += [(before + text[:start], text[start : start + model.context])
                     for start in range(0, len(text), model.context)]
        size += len(tokenizer.decode(row[1:].tolist()).encode("utf-8"))
    bits = -sum(result[0] for result in loglikelihoods(model, requests)) / math.log(2)
    return bits / size


def _evaluate_share(directory, requests_dir, tasks, output_dir, device, batch_size):
    from oe_eval.models.eleuther_huggingface import HFLM_Verbose

    network, config = load(directory, device)
    tokenizer = frame.olmes_tokenizer(os.path.join("data", config["dataset"], "tokenizer.json"))
    wrapper = HFLM_Verbose(pretrained=frame.OlmesModel(network, tokenizer.pad_token_id), tokenizer=tokenizer,
                           max_length=OLMES_LENGTH, batch_size=batch_size, add_bos_token=True, device=device)
    frame.run_olmes(output_dir, wrapper, batch_size, requests_dir, tasks)


def evaluate_run(directory, requests_dir, device="cuda", batch_size=frame.OLMES_BATCH,
                 processes=frame.OLMES_PROCESSES):
    """frame.py's evaluate_run for an H-Net: the reference validation BPB, then the three suites through olmes, with
    the requests from frame.py's build_requests, in processes that share the GPU and the tasks. The scores join the
    run's summary, and olmes's files go to directory/olmes."""
    import multiprocessing

    began = time.time()
    network, _ = load(directory, device)
    val_bpb = validation_bits(byte_language_model(network, device, VALIDATION_BATCH_BYTES), frame.DATA_DIR)
    del network
    if "cuda" in device:
        torch.cuda.empty_cache()
    print(f"val_bpb: {val_bpb:.6f}, {time.time() - began:.0f} seconds", flush=True)
    output_dir = os.path.join(directory, "olmes")
    tasks = frame.olmes_arguments(output_dir, "none")["task"]
    shares = frame.task_shares(tasks, requests_dir, processes)
    share_dirs = [os.path.join(output_dir, f"share-{k}") for k in range(len(shares))]
    context = multiprocessing.get_context("spawn")
    workers = [context.Process(target=_evaluate_share, args=(directory, requests_dir, share, share_dir, device,
                                                             batch_size))
               for share, share_dir in zip(shares, share_dirs)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    if any(worker.exitcode for worker in workers):
        raise RuntimeError("a process that evaluates a share of the tasks failed")
    frame.combine_metrics(output_dir, share_dirs)
    scores = frame.suite_scores(output_dir)
    path = os.path.join(directory, "summary.json")
    summary = json.load(open(path)) if os.path.exists(path) else {}
    summary.update({suite.rsplit(":", 1)[-1]: score for suite, score in scores.items()}, val_bpb=val_bpb,
                   evaluation_seconds=time.time() - began)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print("---")
    for key in ("qa_bpb", "math_bpb", "code_bpb", "val_bpb", "evaluation_seconds"):
        print(f"{key + ':':<20}{summary[key]:.6f}")
    return summary


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__.strip().splitlines()[-1])
    evaluate_run(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "cuda")
