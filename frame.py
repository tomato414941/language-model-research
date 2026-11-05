"""The frame that every run in this repository follows (docs/research/README.md), kept in one place. Only the user
changes it.

The allowed data, the training time, the evaluation and the rule that adopts a candidate. The evaluation is AI2's three
OLMES suites, each the macro average over its tasks of the bits per byte of the correct answer, run by `olmes` at a
pinned revision with its tasks, examples, prompts and scoring unchanged: this file only connects them to the model and
the tokenizer. Validation BPB on the held-out FineWeb data is the yardstick that every run shares, measured unchanged.

A request depends on its task alone, not on the model, and olmes takes hours of CPU to build some tasks' few-shot
prompts, so `python frame.py requests DIR` builds them all beforehand with olmes's own code, and the evaluation hands
them to olmes's runner instead of building them again. `python frame.py count DIR` counts their tokens, and
`python frame.py evaluate OUT_DIR REQUESTS_DIR` evaluates the weights that train.py left. They run where olmes is
installed (olmes.txt), which is not where training runs.
"""

import glob
import json
import math
import os
import shlex
import time
from contextlib import nullcontext
from statistics import mean

import torch

# FineWeb's sample/10BT at revision 9bb295d, as prepare.py tokenizes it. Only the user adds allowed data.
DATA_DIR = os.path.join("data", "fineweb")
# Every set of allowed data: its repository on the Hugging Face Hub, the pinned revision and the folder read. How much
# of each a run reads is free. prepare.py makes FineWeb's files and prepare_extra.py the others', each in data/NAME.
ALLOWED_DATA = {
    "fineweb": ("HuggingFaceFW/fineweb", "9bb295ddab0e05d785b879661af7260fed5140fc", "sample/10BT"),
    "finemath-3plus": ("HuggingFaceTB/finemath", "e92b25a616738fe95dc186b64dfb19f9c8525594", "finemath-3plus"),
    "finemath-4plus": ("HuggingFaceTB/finemath", "e92b25a616738fe95dc186b64dfb19f9c8525594", "finemath-4plus"),
    "stackv2-edu": ("common-pile/stackv2_edu_filtered", "c354dbe88469a1153e97c6a63ac50591849654de", ""),
}
# Seconds of training per run, from scratch; the startup and the first updates, which compile, are not counted.
TIME_BUDGET = 1800
# The updates at the start whose time is not counted, as the loop leaves out its first ten.
UNCOUNTED_UPDATES = 10

OLMES_REVISION = "5a51f502d463b8cdc4a2dcad7d7096c41ff1197e"
OLMES_SUITES = ("olmo3:base_easy:qa_bpb", "olmo3:base_easy:math_bpb", "olmo3:base_easy:code_bpb")
# Rows that each evaluating process scores together; a row of the longest length's logits take about 130 MB.
OLMES_BATCH = 32
# The longest row that olmes's wrapper passes to the model, which must hold the longest answer (2,960 tokens in the
# tokenizer of 2026-11-04); the wrapper cuts longer rows from the start, and OlmesModel reads them in windows.
OLMES_LENGTH = 4096
# Processes that build the requests on the CPU, each taking every few tasks.
OLMES_WORKERS = 5
# Processes that evaluate a run on its GPU, each with a share of the tasks; olmes loads each task's data on the CPU,
# which takes about two seconds a task, while the GPU waits.
OLMES_PROCESSES = 3


def suites_improve(candidate, incumbent):
    """Adopts a candidate when the rates by which its BPB falls below the current best's, averaged over the three
    suites with each counting equally, are positive. Both map each suite to its BPB."""
    rates = [(incumbent[suite] - candidate[suite]) / incumbent[suite] for suite in OLMES_SUITES]
    if not all(math.isfinite(rate) for rate in rates):
        raise ValueError("A decision requires every suite's BPB, finite, for both runs.")
    return mean(rates) > 0


def read_datasets_4_features():
    """Some tasks' data on the Hub were saved by `datasets` 4, which names list features `List`. olmes pins `datasets` 3,
    which calls the same feature `Sequence`; with that name it reads the same examples."""
    from datasets.features import features

    features._FEATURE_TYPES.setdefault("List", features.Sequence)


def olmes_tokenizer(tokenizer_path):
    """A tokenizer that prepare.py trained, as olmes's HuggingFace wrapper takes one: it puts the BOS token, which begins
    every document in training, before each text it encodes, and pads with it."""
    from tokenizers import Tokenizer, processors
    from transformers import PreTrainedTokenizerFast

    from rows import BOS, bos_token

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    bos = bos_token(tokenizer_path)
    tokenizer.post_processor = processors.TemplateProcessing(single=f"{BOS} $A", pair=f"{BOS} $A $B",
                                                             special_tokens=[(BOS, bos)])
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, bos_token=BOS, eos_token=BOS, pad_token=BOS)


class OlmesModel(torch.nn.Module):
    """A model of model.py as olmes's HuggingFace wrapper calls one: the logits of every position under `.logits`.

    The wrapper pads its rows at the end with the BOS token, which no text encodes to, so each row's own length is its
    length without them. A row within the context is read whole. A longer row is read in windows of the context that
    end at the row's end and step back by half of it, each token taking its logits from the window that ends nearest
    after it, so that every token sees at least half a context before it; the last half of the row comes from the
    window that ends with it, as the wrapper's own truncation to the context would give it. olmes reads only the
    logits of the answer at the end of a row, so with `answer_tokens`, at least the longest answer of the task being
    scored, only the windows that reach that far from the end are read."""

    def __init__(self, network, pad):
        from transformers import GPT2Config

        super().__init__()
        self.network = network
        self.pad = pad
        self.answer_tokens = None
        # A causal language model's configuration, which the wrapper reads for its kind.
        self.config = GPT2Config(n_positions=network.config.block_size, vocab_size=network.config.vocab_size)

    @property
    def device(self):
        return next(self.network.parameters()).device

    def tie_weights(self):
        pass

    def read(self, rows):
        autocast = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if rows.is_cuda
                    else nullcontext())
        with autocast:
            return self.network(rows, rows)[0].float()

    def forward(self, input_ids, attention_mask=None, **unused):
        from transformers.modeling_outputs import CausalLMOutput

        context = self.network.config.block_size
        step = context // 2
        rows, length = input_ids.shape
        if length <= context:
            return CausalLMOutput(logits=self.read(input_ids))
        padding = torch.cumprod((input_ids == self.pad).flip(1).int(), dim=1).sum(1)
        lengths = length - padding
        logits = torch.zeros(rows, length, self.network.config.vocab_size, device=input_ids.device)
        within = lengths <= context
        if within.any():
            logits[within, :context] = self.read(input_ids[within, :context].contiguous())
        longer = torch.nonzero(~within).flatten()
        ends = lengths[longer]
        covered = ends.clone()  # each row's positions from here on have their logits
        needed = (ends - self.answer_tokens).clamp(min=0) if self.answer_tokens else torch.zeros_like(ends)
        while (covered > needed).any():
            active = covered > needed
            starts = (ends - context).clamp(min=0)
            windows = starts[active, None] + torch.arange(context, device=input_ids.device)
            read = self.read(torch.gather(input_ids[longer[active]], 1, windows))
            takes = torch.where(starts > 0, ends - step, torch.zeros_like(ends))
            for index, (row, start, take, cover) in enumerate(zip(longer[active].tolist(), starts[active].tolist(),
                                                                  takes[active].tolist(), covered[active].tolist())):
                logits[row, take:cover] = read[index, take - start:cover - start]
            covered = torch.where(active, takes, covered)
            ends = ends - step
        return CausalLMOutput(logits=logits)


def olmes_arguments(output_dir, model_name="none", batch_size=OLMES_BATCH):
    """The arguments of olmes's runner for the three suites, as its launcher makes them from the suites' names."""
    from oe_eval import launch, run_eval

    options = ["--task", *OLMES_SUITES, "--model", model_name, "--output-dir", str(output_dir),
               "--save-raw-requests", "true", "--batch-size", str(batch_size), "--dry-run"]
    if model_name != "none":
        options += ["--model-type", "hf"]
    command = launch.launch_eval(vars(launch._parser.parse_args(options)))["command"]
    words = shlex.split(command)
    assert words[:3] == ["python", "-m", "oe_eval.run_eval"], command
    return vars(run_eval._parser.parse_args(words[3:]))


def task_alias(task_config):
    """The name that olmes's launcher gives a task of a suite, which its aggregation reads."""
    return task_config.get("metadata", {}).get("alias") or task_config["task_name"]


def _build_share(tasks, output_dir):
    from oe_eval import run_eval

    read_datasets_4_features()
    arguments = olmes_arguments(output_dir, "none")
    arguments["task"] = tasks
    run_eval.run_eval(arguments)


def build_requests(requests_dir, workers=OLMES_WORKERS):
    """Builds every request of the three suites with olmes's runner, without a model, in processes that each take every
    few tasks, and keeps each task's requests as requests_dir/ALIAS.jsonl."""
    import multiprocessing
    import shutil

    tasks = olmes_arguments(os.path.join(requests_dir, "unused"), "none")["task"]
    context = multiprocessing.get_context("spawn")
    shares = [(tasks[k::workers], os.path.join(requests_dir, f"worker-{k}")) for k in range(workers)]
    processes = [context.Process(target=_build_share, args=share) for share in shares if share[0]]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    if any(process.exitcode for process in processes):
        raise RuntimeError("a process that builds requests failed")
    built = 0
    for _, directory in shares:
        for metrics_path in glob.glob(os.path.join(directory, "*-metrics.json")):
            alias = task_alias(json.load(open(metrics_path))["task_config"])
            shutil.move(metrics_path.removesuffix("-metrics.json") + "-requests.jsonl",
                        os.path.join(requests_dir, f"{alias}.jsonl"))
            built += 1
        shutil.rmtree(directory)
    if built != len(tasks):
        raise RuntimeError(f"built the requests of {built} of the {len(tasks)} tasks")
    return built


def request_instance(data):
    """A request that olmes wrote, read back as the object that it built. A task that scores the gold answer of a
    generation builds a request that also holds what the generation would need, under the loglikelihood type; olmes's
    own RequestInstance.from_dict reads every loglikelihood request as the plain kind, which takes none of that."""
    from oe_eval.components.instances import RequestInstance
    from oe_eval.components.requests import (GenerateUntilAndLoglikelihoodRequest, GenerateUntilRequest,
                                             LoglikelihoodRequest, LoglikelihoodRollingRequest)

    fields = data["request"]
    if "perplexity_context" in fields:
        kind = GenerateUntilAndLoglikelihoodRequest
    else:
        kind = {"loglikelihood": LoglikelihoodRequest, "generate_until": GenerateUntilRequest,
                "loglikelihood_rolling": LoglikelihoodRollingRequest,
                "generate_until_and_loglikelihood": GenerateUntilAndLoglikelihoodRequest}[data["request_type"]]
    return RequestInstance(request_type=data["request_type"], doc=data["doc"], request=kind(**fields), idx=data["idx"],
                           task_name=data["task_name"], doc_id=data["doc_id"], native_id=data["native_id"],
                           label=data["label"])


def average_arc():
    """olmes groups ARC-Easy and ARC-Challenge as one member of the QA suite without a rule to average them, so its
    aggregation scores neither the group nor the QA suite. The group takes the macro average that every other group
    of the three suites takes."""
    from oe_eval.configs.task_suites import TASK_SUITE_CONFIGS
    from oe_eval.tasks import aggregate_tasks

    name = "arc:rc:bpb::olmes:full"
    if name not in aggregate_tasks.TASK_GROUPS:
        tasks = TASK_SUITE_CONFIGS[name]["tasks"]
        group = {"tasks": tasks, "expected_count": len(tasks), "primary_metric": "macro"}
        aggregate_tasks.TASK_GROUPS = {name: group, **aggregate_tasks.TASK_GROUPS}


def run_olmes(output_dir, wrapper=None, batch_size=OLMES_BATCH, requests_dir=None, tasks=None):
    """Runs the three suites, or the given share of their tasks, through olmes's own runner. With `wrapper`, olmes's
    HuggingFace wrapper around a model, the runner scores the model; without one, it only loads each task's data and
    requests. With `requests_dir`, from build_requests, each task takes its requests from there instead of building
    them."""
    from oe_eval import run_eval

    read_datasets_4_features()
    average_arc()
    arguments = olmes_arguments(output_dir, "none" if wrapper is None else "research", batch_size)
    if tasks is not None:
        arguments["task"] = tasks
    loader, task_loader = run_eval.load_model, run_eval.load_task

    def load_task(task_config, output_dir=None):
        task = task_loader(task_config, output_dir)
        path = os.path.join(requests_dir, f"{task_alias(task.task_config)}.jsonl")

        def read_requests():
            with open(path) as lines:
                task._instances = [request_instance(json.loads(line)) for line in lines]
            if wrapper is not None:
                # A token stands for at least one byte, so an answer has at most as many tokens as bytes.
                answers = [len(instance.request.continuation.encode("utf-8")) for instance in task._instances
                           if getattr(instance.request, "continuation", None)]
                rolling = len(answers) < len(task._instances)
                wrapper.model.answer_tokens = None if rolling or not answers else max(answers) + 1

        task.build_all_requests = read_requests
        return task

    if wrapper is not None:
        run_eval.load_model = lambda model_load_config: wrapper
    if requests_dir is not None:
        run_eval.load_task = load_task
    try:
        return run_eval.run_eval(arguments)
    finally:
        run_eval.load_model, run_eval.load_task = loader, task_loader


def suite_scores(output_dir):
    """Each suite's BPB from olmes's metrics: its primary score, the macro average over its tasks."""
    scores = {}
    with open(os.path.join(output_dir, "metrics-all.jsonl")) as lines:
        for line in lines:
            metrics = json.loads(line)
            alias = metrics.get("task_config", {}).get("metadata", {}).get("alias") or metrics.get("task_name")
            if alias in OLMES_SUITES:
                scores[alias] = metrics["metrics"]["primary_score"]
    missing = [suite for suite in OLMES_SUITES if suite not in scores]
    if missing or not all(math.isfinite(score) for score in scores.values()):
        raise RuntimeError(f"olmes did not score every suite: {missing or scores}")
    return scores


def count_tokens(requests_dir, tokenizer_path, context=1024, workers=OLMES_WORKERS):
    """How many requests the suites make and how many tokens the model reads for them, from build_requests. A row is
    the BOS token, the context and the answer, cut to the model's context as the wrapper cuts it; a rolling request is
    read in windows of the context. Processes share the tasks."""
    from multiprocessing import get_context

    paths = sorted(glob.glob(os.path.join(requests_dir, "*.jsonl")))
    with get_context("spawn").Pool(workers) as pool:
        counted = pool.map(_count_file, [(path, str(tokenizer_path), context) for path in paths])
    return {os.path.basename(path).removesuffix(".jsonl"): count for path, count in zip(paths, counted)}


def _count_file(arguments):
    path, tokenizer_path, context = arguments
    tokenizer = olmes_tokenizer(tokenizer_path)
    requests = tokens = 0
    with open(path) as lines:
        for line in lines:
            instance = json.loads(line)
            request = instance["request"]
            if instance["request_type"] == "loglikelihood":
                whole = len(tokenizer.encode(request["context"] + request["continuation"]))
                tokens += min(whole, context + 1) - 1
            elif instance["request_type"] == "loglikelihood_rolling":
                tokens += len(tokenizer.encode(request["context"]))
            else:
                raise ValueError(f"{path} asks for {instance['request_type']}, which a BPB suite does not")
            requests += 1
    return {"requests": requests, "tokens": tokens}


def validation_bpb(directory, device):
    """The validation BPB as every run has been measured (scoring.py): the first 2^20 tokens of FineWeb's held-out file,
    in rows of the model's context, read by the weights in bfloat16."""
    from tokenizers import Tokenizer

    from loading import nanogpt_model
    from scoring import validation_bits

    config = torch.load(os.path.join(directory, "ckpt.pt"), map_location="cpu")["config"]
    data_dir = os.path.join("data", config["dataset"])
    tokenizer_path = os.path.join(data_dir, "tokenizer.json")
    model = nanogpt_model(directory, tokenizer_path, device, "bfloat16", 262144)
    return validation_bits(model, os.path.join(data_dir, "val.bin"), Tokenizer.from_file(tokenizer_path).decode)


def load_run(directory, device):
    """The last weights that train.py saved in directory, with their COUNT table, and the run's configuration."""
    from model import GPT, GPTConfig

    checkpoint = torch.load(os.path.join(directory, "ckpt.pt"), map_location="cpu")
    network = GPT(GPTConfig(**checkpoint["model_args"]))
    network.load_state_dict({key.removeprefix("_orig_mod."): value for key, value in checkpoint["model"].items()})
    network.to(device)
    if network.config.count:
        from count_table import CountTable

        network.use_count_table(CountTable(checkpoint["config"]["count_table"], device))
    network.eval()
    return network, checkpoint["config"]


def task_shares(tasks, requests_dir, processes):
    """olmes's task configurations split into shares of about equal requests, by the size of their files."""
    sizes = sorted(((os.path.getsize(os.path.join(requests_dir, f"{task_alias(json.loads(task))}.jsonl")), task)
                    for task in tasks), reverse=True)
    shares, totals = [[] for _ in range(processes)], [0] * processes
    for size, task in sizes:
        smallest = totals.index(min(totals))
        shares[smallest].append(task)
        totals[smallest] += size
    return [share for share in shares if share]


def _evaluate_share(directory, requests_dir, tasks, output_dir, device, batch_size):
    from oe_eval.models.eleuther_huggingface import HFLM_Verbose

    network, config = load_run(directory, device)
    tokenizer = olmes_tokenizer(os.path.join("data", config["dataset"], "tokenizer.json"))
    wrapper = HFLM_Verbose(pretrained=OlmesModel(network, tokenizer.pad_token_id), tokenizer=tokenizer,
                           max_length=OLMES_LENGTH, batch_size=batch_size, add_bos_token=True, device=device)
    run_olmes(output_dir, wrapper, batch_size, requests_dir, tasks)


def combine_metrics(output_dir, share_dirs):
    """Every task's metrics from the shares, with the averages that olmes's own aggregation makes of them, written as
    olmes writes metrics-all.jsonl."""
    from oe_eval.tasks.aggregate_tasks import add_aggregate_tasks

    average_arc()
    tasks = []
    for share_dir in share_dirs:
        with open(os.path.join(share_dir, "metrics-all.jsonl")) as lines:
            tasks += [metrics for metrics in map(json.loads, lines) if metrics.get("task_idx") is not None]
    combined = add_aggregate_tasks(tasks) + tasks
    with open(os.path.join(output_dir, "metrics-all.jsonl"), "w") as f:
        for metrics in combined:
            f.write(json.dumps(metrics) + "\n")


def evaluate_run(directory, requests_dir, device="cuda", batch_size=OLMES_BATCH, processes=OLMES_PROCESSES):
    """The evaluation of a run's last weights: the reference validation BPB, then the three suites through olmes, with
    the requests from build_requests, in processes that share the GPU and the tasks. The scores join the run's
    summary, and olmes's files go to directory/olmes."""
    import multiprocessing

    began = time.time()
    val_bpb = validation_bpb(directory, device)
    if "cuda" in device:
        torch.cuda.empty_cache()
    output_dir = os.path.join(directory, "olmes")
    tasks = olmes_arguments(output_dir, "none")["task"]
    shares = task_shares(tasks, requests_dir, processes)
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
    combine_metrics(output_dir, share_dirs)
    scores = suite_scores(output_dir)
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
    import sys

    command = sys.argv[1]
    if command == "requests":
        print(f"built the requests of {build_requests(sys.argv[2])} tasks")
    elif command == "data":
        # every task's data into the Hugging Face cache, with the requests from build_requests, scoring nothing
        run_olmes(sys.argv[2], requests_dir=sys.argv[3])
    elif command == "count":
        counts = count_tokens(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else os.path.join(DATA_DIR, "tokenizer.json"))
        print(json.dumps(counts, indent=1))
        print("total", {key: sum(task[key] for task in counts.values()) for key in ("requests", "tokens")})
    elif command == "evaluate":
        evaluate_run(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "cuda")
    else:
        raise SystemExit("usage: frame.py requests DIR | data OUT REQUESTS_DIR | count REQUESTS_DIR [TOKENIZER] | "
                         "evaluate OUT_DIR REQUESTS_DIR [DEVICE]")
