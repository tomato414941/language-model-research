"""Compare two models' scores question by question; a negative change means the second is better."""

import argparse
import json
import math
import statistics


def compare(first, second):
    benchmarks, languages = {}, {}
    for name, questions in first["questions"].items():
        before, after = questions["bits"], second["questions"][name]["bits"]
        if len(before) != len(after):
            raise ValueError("compare scores of the same questions")
        changes = [new - old for old, new in zip(before, after)]
        benchmarks[name] = {
            "questions": len(changes),
            "change": statistics.fmean(changes),
            "stderr": statistics.stdev(changes) / math.sqrt(len(changes)),
        }
        languages.setdefault(first["benchmarks"][name]["language"], []).append(benchmarks[name])
    return {
        "benchmarks": benchmarks,
        "languages": {
            language: {
                "change": statistics.fmean(row["change"] for row in rows),
                "stderr": math.sqrt(sum(row["stderr"] ** 2 for row in rows)) / len(rows),
            }
            for language, rows in languages.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", help="scoring.py's output for the first model")
    parser.add_argument("second", help="scoring.py's output for the second model")
    arguments = parser.parse_args()
    with open(arguments.first, encoding="utf-8") as first, open(arguments.second, encoding="utf-8") as second:
        print(json.dumps(compare(json.load(first), json.load(second)), indent=2))


if __name__ == "__main__":
    main()
