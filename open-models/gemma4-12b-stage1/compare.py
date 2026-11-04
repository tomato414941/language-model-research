# Compare each run with the base run on the questions both answered: share of unfinished answers and score per task,
# with the standard error of the per-question score difference.
# Arguments: scores/swallow.csv, then the runs to compare (default: every run except base).
import csv, collections, math, statistics, sys

rows = collections.defaultdict(dict)
for r in csv.DictReader(open(sys.argv[1])):
    rows[(r["run"], r["task"])][r["prompt_sha"]] = (float(r["unfinished"]), float(r["score"]))
runs = sys.argv[2:] or sorted({run for run, _ in rows if run != "base"})
for run in runs:
    print(f"== {run}")
    for (r, task), answers in rows.items():
        if r != run:
            continue
        base = rows.get(("base", task), {})
        common = [k for k in answers if k in base]
        if not common:
            print(f"{task:26s} no questions shared with base ({len(answers)} answered)")
            continue
        bu, u = (statistics.fmean(src[k][0] for k in common) for src in (base, answers))
        bs, s = (statistics.fmean(src[k][1] for k in common) for src in (base, answers))
        diff = [answers[k][1] - base[k][1] for k in common]
        se = statistics.stdev(diff) / math.sqrt(len(diff)) if len(diff) > 1 else float("nan")
        print(f"{task:26s} n={len(common):4d} unfinished {bu:.3f} -> {u:.3f} | score {bs:.3f} -> {s:.3f} "
              f"(diff {s - bs:+.3f} +-{se:.3f})")
