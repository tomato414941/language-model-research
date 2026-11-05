"""The frame that every run in this repository follows (docs/research/README.md), kept in one place. Only the user
changes it.

The allowed data, the training time, the evaluation and the rule that adopts a candidate. The evaluation is AI2's three
OLMES suites, each the macro average over its tasks of the bits per byte of the correct answer, with the tasks,
examples, prompts and score formula of `olmes` at a pinned revision. Validation BPB on the held-out FineWeb data is the
yardstick that every run shares, measured unchanged.

olmes asks each question once for every choice, and once more for every choice without the question, but the scores
read only the correct answer after its question. Where olmes is installed (olmes.txt), `python frame.py requests DIR`
builds every request with olmes's own code, which takes hours of CPU for some tasks' few-shot prompts, and
`python frame.py answers REQUESTS_DIR ANSWERS_DIR` keeps the ones that the scores read, with how the suites average
their tasks. `python frame.py evaluate OUT_DIR ANSWERS_DIR` then scores the weights that train.py left without olmes:
the model reads each answer as olmes's HuggingFace wrapper would give it, and the scores follow olmes's formula and
averages. `python frame.py rows ANSWERS_DIR [TOKENIZER]` encodes the answers beforehand for a tokenizer, and
`python frame.py count ANSWERS_DIR [TOKENIZER]` counts the tokens that the model reads for them.
"""

import glob
import hashlib
import itertools
import json
import math
import os
import shlex
import time
from contextlib import nullcontext
from statistics import mean

import numpy as np
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
# The longest row that olmes's wrapper passed to the model when research-v4 was measured, which must hold the longest
# answer (2,960 tokens in the tokenizer of 2026-11-04); longer rows are cut from the start, as the wrapper cut them.
OLMES_LENGTH = 4096
# Tokens that the model reads at once when it scores the answers; their logits take about 2 GB in float32.
OLMES_BATCH_TOKENS = 65536
# Processes that build the requests on the CPU, each taking every few tasks.
OLMES_WORKERS = 5
# olmes's value of log2(e), with which it turns a log-likelihood into bits.
LOG_2_OF_E = 1.44269504089
# What olmes adds to a question's doc_id for the requests that score its choices without the question.
UNCONDITIONED = 1000000


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


def olmes_arguments(output_dir):
    """The arguments of olmes's runner for the three suites without a model, as its launcher makes them from the
    suites' names."""
    from oe_eval import launch, run_eval

    options = ["--task", *OLMES_SUITES, "--model", "none", "--output-dir", str(output_dir), "--save-raw-requests", "true",
               "--dry-run"]
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
    arguments = olmes_arguments(output_dir)
    arguments["task"] = tasks
    run_eval.run_eval(arguments)


def build_requests(requests_dir, workers=OLMES_WORKERS):
    """Builds every request of the three suites with olmes's runner, without a model, in processes that each take every
    few tasks, and keeps each task's requests as requests_dir/ALIAS.jsonl."""
    import multiprocessing
    import shutil

    tasks = olmes_arguments(os.path.join(requests_dir, "unused"))["task"]
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


def scored_requests(requests):
    """The request of each question whose bits per byte its task's score takes, in the order of the questions, from the
    task's requests in the order that olmes wrote them. olmes scores every task of the three suites with MCAccuracy,
    which sets aside the second half of the requests when they ask the choices without their question, groups the rest
    by question and takes the request at the position of the question's label, or the first when the label is not a
    number."""
    asked = requests
    if any(request["doc_id"] >= UNCONDITIONED for request in requests):
        asked, unasked = requests[:len(requests) // 2], requests[len(requests) // 2:]
        if ([(request["doc_id"], request["idx"]) for request in unasked]
                != [(request["doc_id"] + UNCONDITIONED, request["idx"]) for request in asked]):
            raise ValueError("the requests without their question do not follow the ones with it in the same order")
    scored = []
    for _, group in itertools.groupby(sorted(asked, key=lambda request: request["doc_id"]),
                                      key=lambda request: request["doc_id"]):
        group = list(group)
        if any(request["native_id"] != group[0]["native_id"] for request in group):
            raise ValueError("the requests of one question name different questions")
        label = group[0]["label"]
        scored.append(group[0 if label is None or isinstance(label, (str, list)) else label])
    return scored


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


def suite_groups():
    """The groups that olmes's aggregation averages for the three suites, the suites among them, each with its members:
    tasks, or groups averaged the same way. Each takes the macro average of its members."""
    from oe_eval.tasks import aggregate_tasks

    average_arc()
    groups = {}

    def add(name):
        group = aggregate_tasks.TASK_GROUPS[name]
        if group["primary_metric"] != "macro":
            raise ValueError(f"{name} takes the {group['primary_metric']} average, which suite_scores does not")
        groups[name] = list(group["tasks"])
        for member in group["tasks"]:
            if member in aggregate_tasks.TASK_GROUPS:
                add(member)

    for suite in OLMES_SUITES:
        add(suite)
    return groups


def keep_answers(requests_dir, answers_dir):
    """Keeps, from build_requests's files, the request of each question that the scores read, as answers_dir/ALIAS.jsonl
    with the question's doc_id, the context and the answer, and the suites' groups as answers_dir/suites.json."""
    os.makedirs(answers_dir, exist_ok=True)
    tasks = [task_alias(json.loads(task)) for task in olmes_arguments(os.path.join(answers_dir, "unused"))["task"]]
    groups = suite_groups()
    if {member for members in groups.values() for member in members if member not in groups} != set(tasks):
        raise ValueError("the suites' groups do not hold the tasks that olmes runs")
    for path in glob.glob(os.path.join(answers_dir, "rows-*.npz")):
        os.remove(path)  # rows of the answers that these replace
    for alias in tasks:
        with open(os.path.join(requests_dir, f"{alias}.jsonl")) as lines:
            requests = [json.loads(line) for line in lines]
        with open(os.path.join(answers_dir, f"{alias}.jsonl"), "w") as f:
            for request in scored_requests(requests):
                f.write(json.dumps({"doc_id": request["doc_id"], "context": request["request"]["context"],
                                    "continuation": request["request"]["continuation"]}) + "\n")
    with open(os.path.join(answers_dir, "suites.json"), "w") as f:
        json.dump(groups, f, indent=1)
    return len(tasks)


def read_answers(answers_dir):
    """Each task's answers that keep_answers kept, by the task's name."""
    answers = {}
    for path in sorted(glob.glob(os.path.join(answers_dir, "*.jsonl"))):
        with open(path) as lines:
            answers[os.path.basename(path).removesuffix(".jsonl")] = [json.loads(line) for line in lines]
    return answers


def encode_answers(tokenizer_path, requests, length=OLMES_LENGTH):
    """Each answer's tokens as olmes's HuggingFace wrapper makes them, with the BOS token before each text that it
    encodes: as lm_eval does, the context's trailing spaces move to the answer, the answer's tokens are what encoding
    the context and the answer together adds after as many tokens as the context alone has, and a row keeps at most
    `length` + 1 tokens from its end. Returns each row's tokens and how many of the last ones are the answer."""
    from tokenizers import Tokenizer

    from rows import bos_token

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    bos = bos_token(tokenizer_path)
    contexts, wholes = [], []
    for request in requests:
        context, answer = request["context"], request["continuation"]
        if not context:
            raise ValueError("olmes's wrapper reads an answer without a context differently, and no task asks one")
        spaces = len(context) - len(context.rstrip())
        if spaces:
            context, answer = context[:-spaces], context[-spaces:] + answer
        contexts.append(context)
        wholes.append(context + answer)
    rows = []
    # A few thousand at a time: the tokenizer's encodings of all the answers at once take tens of GB.
    for at in range(0, len(requests), 2048):
        for context, whole in zip(tokenizer.encode_batch(contexts[at:at + 2048], add_special_tokens=False),
                                  tokenizer.encode_batch(wholes[at:at + 2048], add_special_tokens=False)):
            answer = whole.ids[len(context.ids):]
            if not 0 < len(answer) <= length:
                raise ValueError(f"an answer of {len(answer)} tokens, which olmes's wrapper refuses")
            rows.append((torch.tensor([bos, *context.ids, *answer][-(length + 1):]), len(answer)))
    return rows


def rows_path(answers_dir, tokenizer_path, length=OLMES_LENGTH):
    """Where make_rows keeps the rows of answers_dir's answers for a tokenizer, named by the tokenizer's contents and the
    length of a row."""
    with open(tokenizer_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()[:16]
    return os.path.join(answers_dir, f"rows-{digest}-{length}.npz")


def make_rows(answers_dir, tokenizer_path, length=OLMES_LENGTH):
    """Encodes every answer of answers_dir beforehand and keeps the rows beside them: encoding takes about eight
    minutes of one core, which a rented machine would otherwise spend."""
    requests = [request for task in read_answers(answers_dir).values() for request in task]
    rows = encode_answers(tokenizer_path, requests, length)
    tokens = torch.cat([row for row, _ in rows]).numpy()
    np.savez(rows_path(answers_dir, tokenizer_path, length),
             tokens=tokens.astype(np.uint16 if tokens.max() < 2**16 else np.int32),
             lengths=np.array([len(row) for row, _ in rows]), answers=np.array([answer for _, answer in rows]))
    return len(rows)


def answer_rows(answers_dir, tokenizer_path, requests, length=OLMES_LENGTH):
    """encode_answers's rows for all the answers of answers_dir in read_answers's order, from make_rows's file for the
    tokenizer when there is one."""
    path = rows_path(answers_dir, tokenizer_path, length)
    if not os.path.exists(path):
        return encode_answers(tokenizer_path, requests, length)
    kept = np.load(path)
    if len(kept["lengths"]) != len(requests):
        raise ValueError(f"{path} holds {len(kept['lengths'])} rows for {len(requests)} answers")
    tokens = torch.from_numpy(kept["tokens"].astype(np.int64))
    return list(zip(torch.split(tokens, kept["lengths"].tolist()), kept["answers"].tolist()))


def answer_windows(length, answer, context):
    """The windows in which the model reads a row of `length` tokens for the predictions at its last `answer` positions,
    as (start, end, first): the model reads the tokens from start to end and gives the predictions from first to end.

    A row within the context is read whole. A longer row is read in windows of the context that end at the row's end and
    step back by half of it, each position taking its prediction from the window that ends nearest after it, so that
    every position sees at least half a context before it; the last half of the row comes from the window that ends
    with it, as the wrapper's own truncation to the context would give it."""
    if length <= context:
        return [(0, length, length - answer)]
    step, needed = context // 2, length - answer
    windows, end = [], length
    while end > needed:
        start = max(end - context, 0)
        first = end - step if start > 0 else 0
        windows.append((start, end, max(first, needed)))
        end = first
    return windows


def read_logits(network, rows):
    """The network's logits at every position of the rows, in float32; on a GPU it computes them in bfloat16."""
    autocast = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if rows.is_cuda
                else nullcontext())
    with autocast:
        return network(rows, rows)[0].float()


@torch.inference_mode()
def answer_log_likelihoods(network, rows, pad, batch_tokens=OLMES_BATCH_TOKENS):
    """The log-likelihood in nats that the network gives each answer of rows from encode_answers: the sum over the
    answer's tokens of the log-softmax of the logits that predict them, which olmes's wrapper takes. The windows of all
    the rows are read longest first, in batches of about batch_tokens tokens padded at the end with `pad`."""
    device = next(network.parameters()).device
    windows = sorted(((index, *window) for index, (tokens, answer) in enumerate(rows)
                      for window in answer_windows(len(tokens) - 1, answer, network.config.block_size)),
                     key=lambda window: window[2] - window[1], reverse=True)
    sums = torch.zeros(len(rows), dtype=torch.float64, device=device)
    at = 0
    while at < len(windows):
        longest = windows[at][2] - windows[at][1]
        batch = windows[at:at + max(1, batch_tokens // longest)]
        at += len(batch)
        inputs = torch.full((len(batch), longest), pad, dtype=torch.long)
        lines, positions, targets, owners = [], [], [], []
        for line, (index, start, end, first) in enumerate(batch):
            tokens = rows[index][0]
            inputs[line, :end - start] = tokens[start:end]
            lines.append(torch.full((end - first,), line))
            positions.append(torch.arange(first - start, end - start))
            targets.append(tokens[first + 1:end + 1])
            owners.append(torch.full((end - first,), index))
        logits = read_logits(network, inputs.to(device))
        picked = logits[torch.cat(lines).to(device), torch.cat(positions).to(device)].log_softmax(-1)
        chosen = picked.gather(1, torch.cat(targets).to(device)[:, None])[:, 0]
        sums.index_add_(0, torch.cat(owners).to(device), chosen.double())
    return sums.tolist()


def task_scores(network, tokenizer_path, answers_dir, length=OLMES_LENGTH, batch_tokens=OLMES_BATCH_TOKENS):
    """Each task's score: the mean over its questions of the bits per byte of the correct answer, by olmes's formula,
    -LOG_2_OF_E times the answer's log-likelihood over its bytes in UTF-8."""
    from rows import bos_token

    answers = read_answers(answers_dir)
    requests = [request for task in answers.values() for request in task]
    totals = iter(answer_log_likelihoods(network, answer_rows(answers_dir, tokenizer_path, requests, length),
                                         bos_token(tokenizer_path), batch_tokens))
    scores = {}
    for name, task in answers.items():
        bits = [-LOG_2_OF_E * (next(totals) / max(len(request["continuation"].encode("utf-8")), 1))
                for request in task]
        scores[name] = sum(bits) / len(bits)
    return scores


def suite_scores(tasks, answers_dir):
    """Each suite's score from its tasks' scores, by the groups in answers_dir/suites.json: a group's score is the mean
    of its members' scores, a member being a task or another group."""
    with open(os.path.join(answers_dir, "suites.json")) as f:
        groups = json.load(f)

    def score(name):
        if name not in groups:
            return tasks[name]
        members = [score(member) for member in groups[name]]
        return sum(members) / len(members)

    scores = {suite: score(suite) for suite in OLMES_SUITES}
    if not all(math.isfinite(value) for value in scores.values()):
        raise RuntimeError(f"a suite has no finite score: {scores}")
    return scores


def count_tokens(answers_dir, tokenizer_path, context=1024):
    """How many answers each task scores and how many tokens the model reads for them, in answer_windows's windows."""
    answers = read_answers(answers_dir)
    rows = iter(answer_rows(answers_dir, tokenizer_path, [request for task in answers.values() for request in task]))
    counts = {}
    for name, task in answers.items():
        mine = [next(rows) for _ in task]
        tokens = sum(end - start for row, answer in mine for start, end, _ in answer_windows(len(row) - 1, answer, context))
        counts[name] = {"answers": len(mine), "tokens": tokens}
    return counts


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


def evaluate_run(directory, answers_dir, device="cuda"):
    """The evaluation of a run's last weights: the reference validation BPB, then the three suites from the answers that
    keep_answers kept. The scores join the run's summary, and each task's score goes to directory/tasks.json."""
    began = time.time()
    val_bpb = validation_bpb(directory, device)
    if "cuda" in device:
        torch.cuda.empty_cache()
    network, config = load_run(directory, device)
    tasks = task_scores(network, os.path.join("data", config["dataset"], "tokenizer.json"), answers_dir)
    scores = suite_scores(tasks, answers_dir)
    with open(os.path.join(directory, "tasks.json"), "w") as f:
        json.dump(tasks, f, indent=1)
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
    elif command == "answers":
        print(f"kept the answers of {keep_answers(sys.argv[2], sys.argv[3])} tasks")
    elif command == "rows":
        print(f"encoded {make_rows(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else os.path.join(DATA_DIR, 'tokenizer.json'))} answers")
    elif command == "count":
        counts = count_tokens(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else os.path.join(DATA_DIR, "tokenizer.json"))
        print(json.dumps(counts, indent=1))
        print("total", {key: sum(task[key] for task in counts.values()) for key in ("answers", "tokens")})
    elif command == "evaluate":
        evaluate_run(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "cuda")
    else:
        raise SystemExit("usage: frame.py requests DIR | answers REQUESTS_DIR ANSWERS_DIR | rows ANSWERS_DIR [TOKENIZER] | "
                         "count ANSWERS_DIR [TOKENIZER] | evaluate OUT_DIR ANSWERS_DIR [DEVICE]")
