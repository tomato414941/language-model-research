"""The frame that every run in this repository follows (docs/research/README.md), kept in one place. Only the user
changes it.

The allowed data, the training time, the evaluation and the rule that adopts a candidate. CORE is nanochat's
evaluation unchanged: its source files at a pinned revision and DCLM's bundle of 22 tasks, checked byte for byte
before they run, pinned as the language-model-autoresearch loop pins them. Validation BPB is only a reference.
"""

import ast
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import tempfile
import time
import urllib.request
import zipfile
from contextlib import nullcontext
from decimal import Decimal

import numpy as np
import torch

# FineWeb's sample/10BT at revision 9bb295d, as prepare.py tokenizes it. Only the user adds allowed data.
DATA_DIR = os.path.join("data", "fineweb")
# Seconds of training per run, from scratch; the startup and the first updates, which compile, are not counted.
TIME_BUDGET = 1800
# The updates at the start whose time is not counted, as the loop leaves out its first ten.
UNCOUNTED_UPDATES = 10

CORE_DIR = os.path.join("data", "core")
CORE_NANOCHAT_REVISION = "92d63d4e8bb4df75c3b71618f31ddde2378b2bcd"
CORE_SOURCE_SHA256 = {
    "nanochat/core_eval.py": "9303997ff17fa48ad5e5730f33b7e79a687cc2a1fdad928666bb63dc3514c51d",
    "scripts/base_eval.py": "644083a99ded710f677b25168ca76fd0416dc7ef3890d346a72f76a20bb522fe",
    "LICENSE": "42bb09f339a8af8b90d3b4969b6f6e4ba4aff3b5c614103b031fccdf71c8f772",
}
CORE_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"
CORE_BUNDLE_SHA256 = "90a7c19e28ee7a52b4f6e1f87658deb9fde7f63deba2379045bdb1fe9ea5d200"
CORE_TASKS = 22


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_core_file(path, expected):
    if not os.path.isfile(path) or _file_sha256(path) != expected:
        raise RuntimeError(f"CORE asset is missing or changed: {path}. Run `python frame.py` without a GPU.")


def _download_core_file(url, path, expected):
    if os.path.exists(path):
        _verify_core_file(path, expected)
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = path + ".tmp"
    try:
        with urllib.request.urlopen(url, timeout=60) as response, open(temp_path, "wb") as f:
            while chunk := response.read(1024 * 1024):
                f.write(chunk)
        _verify_core_file(temp_path, expected)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _core_source_dir(core_dir):
    return os.path.join(core_dir, "nanochat", CORE_NANOCHAT_REVISION)


def verify_core_assets(core_dir=CORE_DIR):
    """The 22 tasks of the bundle, once every source file and every task's data is checked."""
    import yaml

    for name, expected in CORE_SOURCE_SHA256.items():
        _verify_core_file(os.path.join(_core_source_dir(core_dir), name), expected)
    bundle_path = os.path.join(core_dir, "eval_bundle.zip")
    _verify_core_file(bundle_path, CORE_BUNDLE_SHA256)
    with zipfile.ZipFile(bundle_path) as archive:
        tasks = yaml.safe_load(archive.read("eval_bundle/core.yaml"))["icl_tasks"]
        if len(tasks) != CORE_TASKS or len({task["label"] for task in tasks}) != CORE_TASKS:
            raise RuntimeError("CORE requires all 22 upstream tasks.")
        names = ["core.yaml", "eval_meta_data.csv"]
        names.extend("eval_data/" + task["dataset_uri"] for task in tasks)
        for name in names:
            expected = hashlib.sha256(archive.read("eval_bundle/" + name)).hexdigest()
            _verify_core_file(os.path.join(core_dir, "eval_bundle", name), expected)
    return tasks


def prepare_core(core_dir=CORE_DIR):
    """Fetches nanochat's evaluator and its complete official bundle, which needs no GPU."""
    source_url = f"https://raw.githubusercontent.com/karpathy/nanochat/{CORE_NANOCHAT_REVISION}"
    for name, expected in CORE_SOURCE_SHA256.items():
        _download_core_file(f"{source_url}/{name}", os.path.join(_core_source_dir(core_dir), name), expected)
    bundle_path = os.path.join(core_dir, "eval_bundle.zip")
    _download_core_file(CORE_BUNDLE_URL, bundle_path, CORE_BUNDLE_SHA256)
    bundle_dir = os.path.join(core_dir, "eval_bundle")
    if not os.path.exists(bundle_dir):
        with tempfile.TemporaryDirectory(dir=core_dir) as temp_dir:
            with zipfile.ZipFile(bundle_path) as archive:
                archive.extractall(temp_dir)
            os.rename(os.path.join(temp_dir, "eval_bundle"), bundle_dir)
    return verify_core_assets(core_dir)


def _rank_and_world():
    import torch.distributed as dist

    return (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)


def _print0(*arguments, **options):
    if _rank_and_world()[0] == 0:
        print(*arguments, **options)


def _load_nanochat_core(core_dir):
    import yaml

    source_dir = _core_source_dir(core_dir)
    spec = importlib.util.spec_from_file_location("_nanochat_core_eval",
                                                  os.path.join(source_dir, "nanochat", "core_eval.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The original aggregation function, without importing nanochat's training stack. Both files are upstream's,
    # byte for byte, checked before they load.
    runner_path = os.path.join(source_dir, "scripts", "base_eval.py")
    with open(runner_path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=runner_path)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "evaluate_core")
    namespace = dict(os=os, csv=csv, time=time, json=json, random=random, yaml=yaml,
                     get_base_dir=lambda: core_dir, print0=_print0, evaluate_task=module.evaluate_task)
    exec(compile(ast.Module(body=[function], type_ignores=[]), runner_path, "exec"), namespace)
    return namespace["evaluate_core"]


class CoreTokenizer:
    """A tokenizer that prepare.py trained, as nanochat's CORE calls one: on a list of texts, with a token first."""

    def __init__(self, path):
        from tokenizers import Tokenizer

        from rows import bos_token

        self.tokenizer = Tokenizer.from_file(str(path))
        self.bos = bos_token(path)

    def get_bos_token_id(self):
        return self.bos

    def __call__(self, texts, prepend=None):
        single = isinstance(texts, str)
        encoded = [encoding.ids for encoding in
                   self.tokenizer.encode_batch([texts] if single else list(texts), add_special_tokens=False)]
        if prepend is not None:
            encoded = [[prepend, *ids] for ids in encoded]
        return encoded[0] if single else encoded


class CoreModel:
    """A model of model.py as nanochat's CORE calls one: the logits of every position, and its context."""

    def __init__(self, network):
        self.network = network
        self.max_seq_len = network.config.block_size

    def __call__(self, input_ids):
        # Given targets, the model returns the logits of every position rather than the last one's.
        return self.network(input_ids, input_ids)[0]


@torch.no_grad()
def evaluate_core(network, tokenizer, device, core_dir=CORE_DIR):
    """Runs nanochat's CORE once, with all examples, the original prompts and the original scoring."""
    tasks = verify_core_assets(core_dir)
    evaluator = _load_nanochat_core(core_dir)
    # Eager execution, so that no graph is compiled for every prompt.
    network = getattr(network, "_orig_mod", network)
    training = network.training
    network.eval()
    try:
        result = evaluator(CoreModel(network), tokenizer, device, max_per_task=-1)
    finally:
        network.train(training)
    labels = {task["label"] for task in tasks}
    if set(result["results"]) != labels or set(result["centered_results"]) != labels:
        raise RuntimeError("CORE did not finish all 22 tasks.")
    scores = [result["core_metric"], *result["results"].values(), *result["centered_results"].values()]
    if not all(math.isfinite(score) for score in scores):
        raise RuntimeError("CORE produced a non-finite score.")
    return result


def save_core_results(result, directory):
    """Keeps nanochat's result dictionary and its own per-task CSV."""
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "core.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    with open(os.path.join(directory, "core.csv"), "w", encoding="utf-8", newline="") as f:
        f.write(f"{'Task':<35}, {'Accuracy':<10}, {'Centered':<10}\n")
        for label in result["results"]:
            f.write(f"{label:<35}, {result['results'][label]:<10.6f}, {result['centered_results'][label]:<10.6f}\n")
        f.write(f"{'CORE':<35}, {'':<10}, {result['core_metric']:<10.6f}\n")


def core_improves(candidate, incumbent):
    """Adopts a candidate whose unrounded CORE is higher than the current best's."""
    if not math.isfinite(candidate) or not math.isfinite(incumbent):
        raise ValueError("A CORE decision requires two finite scores.")
    return Decimal(str(candidate)) > Decimal(str(incumbent))


def token_bytes(tokenizer_path):
    """How many bytes of text each token stands for. In byte-level BPE each character of a token is one byte;
    the BOS token stands for none."""
    from tokenizers import Tokenizer

    from rows import bos_token

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    sizes = np.array([len(tokenizer.id_to_token(i)) for i in range(tokenizer.get_vocab_size())], dtype=np.int64)
    sizes[bos_token(tokenizer_path)] = 0
    return sizes


@torch.no_grad()
def evaluate_bpb(network, validation_path, tokenizer_path, device, rows_per_pass=64):
    """The reference validation BPB: rows of the model's context and one more token, each beginning at the head of
    a document and none overlapping, over the whole validation file, as training reads its rows. Processes under
    torchrun share the rows."""
    import torch.distributed as dist

    rank, world = _rank_and_world()
    nats, size = bpb_sums(network, validation_path, tokenizer_path, device, rank, world, rows_per_pass)
    if world > 1:
        # gloo, which lets the processes share a GPU, sums on the host
        nats, size = nats.cpu(), size.cpu()
        dist.all_reduce(nats)
        dist.all_reduce(size)
    return (nats / (math.log(2) * size)).item()


@torch.no_grad()
def bpb_sums(network, validation_path, tokenizer_path, device, rank=0, world=1, rows_per_pass=64):
    """The nats of the counted targets and their bytes over every world-th row of evaluate_bpb, from rank."""
    from rows import bos_token

    network = getattr(network, "_orig_mod", network)
    data = np.fromfile(validation_path, dtype=np.uint16)
    length = network.config.block_size + 1
    heads = np.flatnonzero(data == bos_token(tokenizer_path))
    starts, next_free = [], 0
    for head in heads:
        if head >= next_free and head + length <= len(data):
            starts.append(head)
            next_free = head + length - 1
    starts = starts[rank::world]
    sizes = torch.from_numpy(token_bytes(tokenizer_path)).to(device)
    training = network.training
    network.eval()
    nats = torch.zeros((), dtype=torch.float64, device=device)
    size = torch.zeros((), dtype=torch.int64, device=device)
    try:
        for first in range(0, len(starts), rows_per_pass):
            rows = np.stack([data[start:start + length] for start in starts[first:first + rows_per_pass]])
            rows = torch.from_numpy(rows.astype(np.int64)).to(device)
            inputs, targets = rows[:, :-1].contiguous(), rows[:, 1:].contiguous()
            logits, _ = network(inputs, targets)
            losses = torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), targets.flatten(),
                                                       reduction="none")
            counted = sizes[targets.flatten()]
            nats += (losses * (counted > 0)).sum().double()
            size += counted.sum()
    finally:
        network.train(training)
    return nats, size.double()


def evaluate_run(directory, device="cuda"):
    """The evaluation of a run, from the last weights that train.py saved in directory: the reference validation
    BPB, then CORE. Under torchrun, its processes share one GPU and split the rows and CORE's examples, as nanochat
    splits them over several GPUs; each example is scored as it would be alone. The results go beside the weights."""
    import torch.distributed as dist

    from model import GPT, GPTConfig

    began = time.time()
    if "RANK" in os.environ:
        # NCCL takes one GPU per process; gloo lets several share one.
        dist.init_process_group("gloo")
    rank, _ = _rank_and_world()
    checkpoint = torch.load(os.path.join(directory, "ckpt.pt"), map_location="cpu")
    network = GPT(GPTConfig(**checkpoint["model_args"]))
    network.load_state_dict({key.removeprefix("_orig_mod."): value for key, value in checkpoint["model"].items()})
    network.to(device)
    if network.config.count:
        from count_table import CountTable

        network.use_count_table(CountTable(checkpoint["config"]["count_table"], device))
    data_dir = os.path.join("data", checkpoint["config"]["dataset"])
    tokenizer_path = os.path.join(data_dir, "tokenizer.json")
    context = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if "cuda" in device else nullcontext()
    with context:
        val_bpb = evaluate_bpb(network, os.path.join(data_dir, "val.bin"), tokenizer_path, device)
        core = evaluate_core(network, CoreTokenizer(tokenizer_path), device)
    if rank == 0:
        save_core_results(core, directory)
        path = os.path.join(directory, "summary.json")
        with open(path) as f:
            summary = json.load(f)
        summary.update(val_bpb=val_bpb, core_metric=core["core_metric"], evaluation_seconds=time.time() - began)
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        print("---")
        print(f"core_metric:      {core['core_metric']:.17g}")
        print(f"val_bpb:          {val_bpb:.6f}")
        print(f"evaluation_seconds: {summary['evaluation_seconds']:.1f}")
    if dist.is_initialized():
        dist.destroy_process_group()
    return core, val_bpb


if __name__ == "__main__":
    import sys

    if sys.argv[1:2] == ["evaluate"]:
        evaluate_run(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "cuda")
    else:
        print(f"CORE: all {len(prepare_core())} upstream tasks ready (nanochat {CORE_NANOCHAT_REVISION})")
