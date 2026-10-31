"""Score a language model on published benchmarks, through the LM Evaluation Harness.

Every benchmark here asks a model to choose among continuations, so a model only has to give the
probability of a continuation after a context, whatever it is made of. Besides the accuracy, each
benchmark reports the bits per byte of its correct choices, which moves with every gain in the
answer's probability, so small models show progress before their answers change; see DataDecide
(arXiv:2504.11393) and Signal and Noise (arXiv:2508.13144).

Finding whether a change helped needs far fewer questions than a benchmark holds, so by default each
benchmark asks a fixed sample, and each question's result is kept for comparing models question by
question.
"""

import argparse
import json
import math
import random
import statistics
from importlib.metadata import version
from pathlib import Path

import torch
from lm_eval import evaluator
from lm_eval.api.model import TemplateLM
from lm_eval.api.task import ConfigurableTask
from lm_eval.tasks import TaskManager
from lm_eval.tasks._yaml_loader import load_yaml

# Each benchmark's language and the accuracy read from it. OLMES (arXiv:2406.08446) scores
# HellaSwag, ARC-Easy, PIQA and MMLU's cloze form by each choice's loglikelihood per character, the
# harness's acc_norm, and so is ARC-Challenge here; the others use plain accuracy.
BENCHMARKS = {
    "mmlu_continuation": ("en", "acc_norm"),
    "arc_easy": ("en", "acc_norm"),
    "arc_challenge": ("en", "acc_norm"),
    "hellaswag": ("en", "acc_norm"),
    "piqa": ("en", "acc_norm"),
    "sciq": ("en", "acc"),
    "ja_leaderboard_xwinograd": ("ja", "acc"),
    "ja_leaderboard_jcommonsenseqa": ("ja", "acc"),
    "ja_leaderboard_jnli": ("ja", "acc"),
}
JGLUE_URL = (
    "https://raw.githubusercontent.com/yahoojapan/JGLUE/refs/tags/v1.2.0/"
    "datasets/{name}-v1.2/{split}-v1.2.json"
)
JGLUE_TASKS = {
    "ja_leaderboard_jcommonsenseqa": "jcommonsenseqa",
    "ja_leaderboard_jnli": "jnli",
}
# JGLUE's loader numbered JNLI's labels in this order, which the task's choices follow.
JNLI_LABELS = ("entailment", "contradiction", "neutral")


def number_jnli_labels(dataset):
    return dataset.map(lambda doc: {"label": JNLI_LABELS.index(doc["label"])})


def jglue_task(manager, name):
    """Read JGLUE's original JSON directly: datasets 4+ cannot run its loader script."""
    path = manager.task_index[name].yaml_path
    config = load_yaml(path, resolve_func=True)
    # Keep the published prompts, validation split, examples and metrics.
    config.update(
        dataset_path="json",
        dataset_name=None,
        dataset_kwargs={
            "data_files": {
                split: JGLUE_URL.format(name=JGLUE_TASKS[name], split=file)
                for split, file in (("train", "train"), ("validation", "valid"))
            }
        },
    )
    if name == "ja_leaderboard_jnli":
        config["process_docs"] = number_jnli_labels
    config["metadata"]["config_source"] = str(path)
    return ConfigurableTask(config=config)


class ContinuationModel(TemplateLM):
    """Gives the harness the probability of a continuation after a context.

    `logits` maps a batch of token rows to each position's logits, `encode` a string to its tokens,
    `context` is the most tokens a row may hold, and `prefix` the token that begins a document.
    """

    def __init__(self, logits, encode, context, prefix, device="cpu", batch_tokens=4096):
        super().__init__()
        self.logits = logits
        self.encode = encode
        self.context = context
        self.prefix = prefix
        self._device = device
        self.batch_tokens = batch_tokens

    @property
    def eot_token_id(self):
        return self.prefix

    @property
    def prefix_token_id(self):
        return self.prefix

    @property
    def tokenizer_name(self):
        return "continuation-model"

    def tok_encode(self, string, add_special_tokens=None, **kwargs):
        tokens = list(self.encode(string))
        if add_special_tokens is not False and (not tokens or tokens[0] != self.prefix):
            tokens.insert(0, self.prefix)
        return tokens

    @torch.inference_mode()
    def _loglikelihood_tokens(self, requests, disable_tqdm=False, **kwargs):
        prepared = []
        for index, (key, context, continuation) in enumerate(requests):
            if not context or not 0 < len(continuation) <= self.context:
                raise ValueError("each continuation must fit within the context length")
            tokens = (context + continuation)[-(self.context + 1) :]
            prepared.append((index, key, tokens, len(continuation)))
        # Rows of like length share a batch, and padding after a row's end cannot reach it.
        prepared.sort(key=lambda request: -len(request[2]))
        results = [None] * len(requests)
        start = 0
        while start < len(prepared):
            width = len(prepared[start][2]) - 1
            batch = prepared[start : start + max(1, self.batch_tokens // width)]
            start += len(batch)
            inputs = torch.full((len(batch), width), self.prefix, dtype=torch.long)
            for row, (_, _, tokens, _) in enumerate(batch):
                inputs[row, : len(tokens) - 1] = torch.tensor(tokens[:-1])
            # Summed in 32 bits: in bfloat16 the sums of two close choices can come out equal,
            # and a tie goes to the first choice.
            scores = self.logits(inputs.to(self.device)).float().log_softmax(-1)
            for row, (index, key, tokens, length) in enumerate(batch):
                end = len(tokens) - 1
                targets = torch.tensor(tokens[-length:], device=self.device)
                chosen = scores[row, end - length : end]
                result = (
                    chosen.gather(-1, targets[:, None]).sum().item(),
                    bool((chosen.argmax(-1) == targets).all().item()),
                )
                results[index] = result
                if key is not None:
                    self.cache_hook.add_partial("loglikelihood", key, result)
        return results

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        raise NotImplementedError("the benchmarks here only compare choices")

    def generate_until(self, requests, disable_tqdm=False):
        raise NotImplementedError("the benchmarks here only compare choices")


def bits_per_byte(context, continuation, loglikelihood):
    # The harness scores the context's trailing spaces with the continuation.
    scored = context[len(context.rstrip()) :] + continuation
    return -loglikelihood / (math.log(2) * len(scored.encode("utf-8")))


def correct_choice(task, sample):
    """The index of the choice that the harness counts as correct."""
    doc = sample["doc"]
    # The harness finds the correct choice this way when it scores the choices.
    gold = task.doc_to_text(doc) if task.multiple_input else task.doc_to_target(doc)
    if isinstance(gold, str):
        gold = task.doc_to_choice(doc).index(gold)
    return gold


def answer_bits(task, sample):
    gold = correct_choice(task, sample)
    context, continuation = sample["arguments"][gold]
    return bits_per_byte(context, continuation, sample["filtered_resps"][gold][0])


def choose_questions(sizes, questions):
    """The same questions every time: a sample shared among a benchmark's tasks by their sizes."""
    total = sum(sizes.values())
    return {
        task: sorted(random.Random(task).sample(range(size), min(size, max(1, round(questions * size / total)))))
        for task, size in sizes.items()
    }


def summarize(rows_by_benchmark):
    """Each benchmark's scores over its questions, and each language's mean over its benchmarks.

    `rows_by_benchmark` holds each question's accuracy and correct answer's bits per byte, in an
    order that every model shares.
    """
    benchmarks, languages = {}, {}
    for name, rows in rows_by_benchmark.items():
        language, metric = BENCHMARKS[name]
        benchmarks[name] = {
            "language": language,
            "metric": metric,
            "items": len(rows),
            "accuracy": statistics.fmean(row["accuracy"] for row in rows),
            "gold_bits_per_byte": statistics.fmean(row["bits"] for row in rows),
        }
        languages.setdefault(language, []).append(benchmarks[name])
    return {
        "questions": {
            name: {
                "correct": [row["accuracy"] for row in rows],
                "bits": [round(row["bits"], 6) for row in rows],
            }
            for name, rows in rows_by_benchmark.items()
        },
        "benchmarks": benchmarks,
        "languages": {
            language: {
                "accuracy": statistics.fmean(row["accuracy"] for row in rows),
                "gold_bits_per_byte": statistics.fmean(row["gold_bits_per_byte"] for row in rows),
            }
            for language, rows in languages.items()
        },
    }


def score(model, names=None, questions=None):
    """Ask a model the benchmarks: each one's whole set, or that many questions from it."""
    names = list(BENCHMARKS if names is None else names)
    manager = TaskManager()
    loaded = manager.load([jglue_task(manager, name) if name in JGLUE_TASKS else name for name in names])
    chosen = [loaded["groups"][name] if name in loaded["groups"] else loaded["tasks"][name] for name in names]
    members = {
        name: [task.task_name for task in loaded["groups"][name].get_all_tasks()]
        if name in loaded["groups"]
        else [name]
        for name in names
    }
    samples = None
    if questions is not None:
        samples = {}
        for name in names:
            sizes = {member: len(loaded["tasks"][member].eval_docs) for member in members[name]}
            samples.update(choose_questions(sizes, questions))
    result = evaluator.simple_evaluate(
        model=model, tasks=chosen, task_manager=manager, samples=samples, log_samples=True
    )
    rows = {}
    for name in names:
        metric = BENCHMARKS[name][1]
        rows[name] = [
            {"accuracy": sample[metric], "bits": answer_bits(loaded["tasks"][member], sample)}
            for member in members[name]
            for sample in sorted(result["samples"][member], key=lambda sample: sample["doc_id"])
        ]
    return {
        **summarize(rows),
        "scope": "full" if questions is None else "sample",
        "harness": version("lm_eval"),
    }


def huggingface_model(path, device, precision, batch_tokens):
    """A model in Hugging Face's format, with the tokenizer saved beside it."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    network = AutoModelForCausalLM.from_pretrained(path, dtype=getattr(torch, precision)).to(device)
    network.eval()
    return ContinuationModel(
        logits=lambda tokens: network(tokens).logits,
        encode=lambda text: tokenizer.encode(text, add_special_tokens=False),
        context=network.config.max_position_embeddings,
        prefix=tokenizer.bos_token_id,
        device=device,
        batch_tokens=batch_tokens,
    )


def nanogpt_model(directory, tokenizer_path, device, precision, batch_tokens):
    """A checkpoint that train.py saved, with the tokenizer that prepare.py trained."""
    from tokenizers import Tokenizer

    from model import GPT, GPTConfig
    from prepare import BOS

    checkpoint = torch.load(Path(directory) / "ckpt.pt", map_location=device)
    network = GPT(GPTConfig(**checkpoint["model_args"]))
    network.load_state_dict({key.removeprefix("_orig_mod."): value for key, value in checkpoint["model"].items()})
    network.to(device, dtype=getattr(torch, precision))
    network.eval()
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return ContinuationModel(
        # Given targets, nanoGPT's model returns every position's logits rather than the last one's.
        logits=lambda tokens: network(tokens, tokens)[0],
        encode=lambda text: tokenizer.encode(text, add_special_tokens=False).ids,
        context=network.config.block_size,
        prefix=tokenizer.token_to_id(BOS),
        device=device,
        batch_tokens=batch_tokens,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", help="train.py's output directory, or a Hugging Face model")
    parser.add_argument("--tokenizer", help="prepare.py's tokenizer, which marks the model as nanoGPT's")
    parser.add_argument("--output", required=True, help="where to write the scores, as JSON")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="bfloat16")
    parser.add_argument("--batch-tokens", type=int, default=262144)
    parser.add_argument("--language", choices=["en", "ja"], default="en")
    parser.add_argument("--questions", type=int, default=200, help="from each benchmark; 0 for all of them")
    arguments = parser.parse_args()
    load = [arguments.device, arguments.precision, arguments.batch_tokens]
    model = (
        nanogpt_model(arguments.model, arguments.tokenizer, *load)
        if arguments.tokenizer
        else huggingface_model(arguments.model, *load)
    )
    names = [name for name, (language, _) in BENCHMARKS.items() if language == arguments.language]
    result = score(model, names, arguments.questions or None)
    with open(arguments.output, "w", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
