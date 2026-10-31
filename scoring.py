"""Score a model in seconds: the bits per byte of published benchmarks' correct answers.

A model's probability of the correct answer moves with every gain, so small models show progress
before their answers change; see DataDecide (arXiv:2504.11393) and Signal and Noise
(arXiv:2508.13144). That needs neither the wrong choices nor the harness, which harness.py runs only
to write these questions out and to score a model in full at a milestone. Each question's result is
kept for comparing models question by question. The validation text's bits per byte is reported
beside them as a monitor; it comes from the training data's own distribution.
"""

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np

from loading import encode_pair, huggingface_model, loglikelihoods, nanogpt_model


def bits_per_byte(context, continuation, loglikelihood):
    # The harness scores the context's trailing spaces with the continuation.
    scored = context[len(context.rstrip()) :] + continuation
    return -loglikelihood / (math.log(2) * len(scored.encode("utf-8")))


def read_questions(path):
    with open(path, encoding="utf-8") as source:
        return [json.loads(line) for line in source]


def summarize(questions, bits):
    """Each benchmark's mean over its questions, and each language's mean over its benchmarks."""
    by_benchmark = {}
    for question, value in zip(questions, bits):
        by_benchmark.setdefault(question["benchmark"], {"language": question["language"], "bits": []})
        by_benchmark[question["benchmark"]]["bits"].append(value)
    benchmarks, languages = {}, {}
    for name, row in by_benchmark.items():
        benchmarks[name] = {
            "language": row["language"],
            "questions": len(row["bits"]),
            "gold_bits_per_byte": statistics.fmean(row["bits"]),
        }
        languages.setdefault(row["language"], []).append(benchmarks[name]["gold_bits_per_byte"])
    return {
        "questions": {name: {"bits": [round(value, 6) for value in row["bits"]]} for name, row in by_benchmark.items()},
        "benchmarks": benchmarks,
        "languages": {language: {"gold_bits_per_byte": statistics.fmean(rows)} for language, rows in languages.items()},
    }


def score_questions(model, questions):
    requests = [encode_pair(model, question["context"], question["answer"]) for question in questions]
    results = loglikelihoods(model, requests)
    return summarize(
        questions,
        [
            bits_per_byte(question["context"], question["answer"], result[0])
            for question, result in zip(questions, results)
        ],
    )


def validation_bits(model, path, decode, tokens=2**20):
    """The bits per byte of the validation text's first tokens, read in rows of the model's context."""
    data = np.fromfile(path, dtype=np.uint16, count=tokens).tolist()
    rows = [data[start : start + model.context + 1] for start in range(0, len(data) - 1, model.context)]
    rows = [row for row in rows if len(row) > 1]
    results = loglikelihoods(model, [(row[:1], row[1:]) for row in rows])
    size = sum(len(decode(row[1:]).encode("utf-8")) for row in rows)
    return -sum(result[0] for result in results) / (math.log(2) * size)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", help="train.py's output directory, or a Hugging Face model")
    parser.add_argument("--questions", required=True, help="the file that harness.py wrote out")
    parser.add_argument("--output", required=True, help="where to write the scores, as JSON")
    parser.add_argument("--tokenizer", help="prepare.py's tokenizer, which marks the model as nanoGPT's")
    parser.add_argument("--validation", help="prepare.py's val.bin, read with that tokenizer")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="bfloat16")
    parser.add_argument("--batch-tokens", type=int, default=262144)
    arguments = parser.parse_args()
    load = [arguments.device, arguments.precision, arguments.batch_tokens]
    model = (
        nanogpt_model(arguments.model, arguments.tokenizer, *load)
        if arguments.tokenizer
        else huggingface_model(arguments.model, *load)
    )
    result = score_questions(model, read_questions(arguments.questions))
    if arguments.validation:
        from tokenizers import Tokenizer

        decode = Tokenizer.from_file(arguments.tokenizer).decode
        result["validation_bits_per_byte"] = validation_bits(model, arguments.validation, decode)
    Path(arguments.output).write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "questions"}, indent=2))


if __name__ == "__main__":
    main()
