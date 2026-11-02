"""Run the LM Evaluation Harness: to write benchmarks' questions out, and to score a trained model in full.

Every benchmark here asks a model to choose among continuations, so a model only has to give the
probability of a continuation after a context. Beside scoring.py's bits per byte of the correct
answers, a full score here reports each benchmark's accuracy over all its questions, and keeps each
question's result for comparing models question by question.
"""

import argparse
import json
import statistics
from importlib.metadata import version
from pathlib import Path

from lm_eval import evaluator
from lm_eval.api.model import TemplateLM
from lm_eval.api.task import ConfigurableTask
from lm_eval.tasks import TaskManager
from lm_eval.tasks._yaml_loader import load_yaml

from loading import LanguageModel, huggingface_model, loglikelihoods, nanogpt_model
from scoring import bits_per_byte

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
# MMLU in the form its authors asked it (arXiv:2009.03300): five examples from the subject's development
# set before each question, answered by letter. The harness asks with no examples unless told to. It is
# scored only when asked for and kept apart from the benchmarks above, among which mmlu_continuation
# already asks the same questions.
AUTHORS_MMLU = {"mmlu": ("en", "acc")}
AUTHORS_MMLU_EXAMPLES = 5
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


class HarnessModel(TemplateLM):
    """Gives the harness a loaded model's probability of a continuation after a context."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self._device = model.device

    @property
    def eot_token_id(self):
        return self.model.prefix

    @property
    def prefix_token_id(self):
        return self.model.prefix

    @property
    def tokenizer_name(self):
        return "continuation-model"

    def tok_encode(self, string, add_special_tokens=None, **kwargs):
        tokens = list(self.model.encode(string))
        if add_special_tokens is not False and (not tokens or tokens[0] != self.model.prefix):
            tokens.insert(0, self.model.prefix)
        return tokens

    def _loglikelihood_tokens(self, requests, disable_tqdm=False, **kwargs):
        results = loglikelihoods(self.model, [(context, continuation) for _, context, continuation in requests])
        for (key, _, _), result in zip(requests, results):
            if key is not None:
                self.cache_hook.add_partial("loglikelihood", key, result)
        return results

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        raise NotImplementedError("the benchmarks here only compare choices")

    def generate_until(self, requests, disable_tqdm=False):
        raise NotImplementedError("the benchmarks here only compare choices")


class BlankModel(HarnessModel):
    """Answers nothing, for reading the requests that the harness writes for its questions."""

    def __init__(self):
        super().__init__(
            LanguageModel(logits=None, encode=lambda text: list(text.encode("utf-8")), context=2**20, prefix=256)
        )

    def _loglikelihood_tokens(self, requests, disable_tqdm=False, **kwargs):
        return [(0.0, False)] * len(requests)


def correct_choice(task, sample):
    """The index of the choice that the harness counts as correct."""
    doc = sample["doc"]
    # The harness finds the correct choice this way when it scores the choices.
    gold = task.doc_to_text(doc) if task.multiple_input else task.doc_to_target(doc)
    if isinstance(gold, str):
        gold = task.doc_to_choice(doc).index(gold)
    return gold


def ask(model, names, limit=None, examples=None):
    """Ask the benchmarks and yield each question with the task that asked it."""
    manager = TaskManager()
    loaded = manager.load([jglue_task(manager, name) if name in JGLUE_TASKS else name for name in names])
    chosen = [loaded["groups"][name] if name in loaded["groups"] else loaded["tasks"][name] for name in names]
    result = evaluator.simple_evaluate(
        model=model, tasks=chosen, task_manager=manager, limit=limit, num_fewshot=examples, log_samples=True
    )
    for name in names:
        members = (
            [task.task_name for task in loaded["groups"][name].get_all_tasks()]
            if name in loaded["groups"]
            else [name]
        )
        for member in members:
            task = loaded["tasks"][member]
            for sample in sorted(result["samples"][member], key=lambda sample: sample["doc_id"]):
                yield name, member, task, sample


def write_questions(path, names):
    """Write each question's context and correct answer, as the harness asks them."""
    with open(path, "w", encoding="utf-8") as output:
        for name, member, task, sample in ask(BlankModel(), names):
            context, answer = sample["arguments"][correct_choice(task, sample)]
            row = {
                "benchmark": name,
                "language": BENCHMARKS[name][0],
                "task": member,
                "doc": sample["doc_id"],
                "context": context,
                "answer": answer,
            }
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def score_in_full(model, benchmarks, limit=None, examples=None):
    """Each benchmark's accuracy and correct answers' bits per byte, as the harness scores them."""
    rows = {name: [] for name in benchmarks}
    for name, _, task, sample in ask(HarnessModel(model), list(benchmarks), limit, examples):
        gold = correct_choice(task, sample)
        context, answer = sample["arguments"][gold]
        rows[name].append(
            {
                "accuracy": sample[benchmarks[name][1]],
                "bits": bits_per_byte(context, answer, sample["filtered_resps"][gold][0]),
            }
        )
    return {
        "questions": {
            name: {
                "correct": [item["accuracy"] for item in row],
                "bits": [round(item["bits"], 6) for item in row],
            }
            for name, row in rows.items()
        },
        "benchmarks": {
            name: {
                "language": benchmarks[name][0],
                "metric": benchmarks[name][1],
                "questions": len(row),
                "accuracy": statistics.fmean(item["accuracy"] for item in row),
                "gold_bits_per_byte": statistics.fmean(item["bits"] for item in row),
            }
            for name, row in rows.items()
        },
        "scope": "full" if limit is None else "sample",
        "harness": version("lm_eval"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", help="where to write the questions or the score")
    parser.add_argument("--language", choices=["en", "ja"], default="en")
    parser.add_argument("--model", help="score this model in full, rather than writing the questions out")
    parser.add_argument("--tokenizer", help="prepare.py's tokenizer, which marks the model as nanoGPT's")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="bfloat16")
    parser.add_argument("--batch-tokens", type=int, default=262144)
    parser.add_argument("--limit", type=int, help="ask only this many questions of each task, to try it")
    parser.add_argument("--authors-mmlu", action="store_true", help="score MMLU in its authors' form instead")
    arguments = parser.parse_args()
    benchmarks = {name: row for name, row in BENCHMARKS.items() if row[0] == arguments.language}
    if not arguments.model:
        write_questions(arguments.output, list(benchmarks))
        return
    load = [arguments.device, arguments.precision, arguments.batch_tokens]
    model = (
        nanogpt_model(arguments.model, arguments.tokenizer, *load)
        if arguments.tokenizer
        else huggingface_model(arguments.model, *load)
    )
    score = (
        score_in_full(model, AUTHORS_MMLU, arguments.limit, AUTHORS_MMLU_EXAMPLES)
        if arguments.authors_mmlu
        else score_in_full(model, benchmarks, arguments.limit)
    )
    Path(arguments.output).write_text(json.dumps(score, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
