# Turn the raw Stage 1 outputs into the tables kept in scores/: one row per question for the Swallow tasks and one
# row per conversation for tau2. The raw outputs hold the benchmark questions and the model answers, so they stay
# out of git; the tables keep only a hash of each prompt, the scores and how each answer or conversation ended.
# Arguments: raw directory (swallow/<run>/ with lighteval details, tau2/<run>/results.json), output directory.
import csv, glob, hashlib, json, os, statistics, sys
import pyarrow.parquet as pq

# The metric each task's score is read from; other tasks fall back to extractive_match, then to the first metric.
METRIC = {"aime_N4:24": "extractive_match_pass@1:4", "aime_N4:25": "extractive_match_pass@1:4",
          "mifeval_ja": "inst_level_strict_acc", "ifbench_singleturn": "inst_level_strict_acc"}


def mean(values):
    return statistics.fmean(float(v) for v in values)


def score_of(metrics, task):
    for key in (METRIC.get(task), "extractive_match"):
        if key and metrics.get(key) is not None:
            value = metrics[key]
            break
    else:
        value = next(v for v in metrics.values() if v is not None)
    if isinstance(value, dict):
        value = mean(v for v in value.values() if v is not None)
    if isinstance(value, (list, tuple)):
        value = mean(value)
    return float(value)


def swallow_rows(raw):
    for run_dir in sorted(glob.glob(os.path.join(raw, "swallow", "*"))):
        run = os.path.basename(run_dir)
        for path in sorted(glob.glob(os.path.join(run_dir, "**", "*.parquet"), recursive=True)):
            task = os.path.basename(path).split("|")[1]
            for r in pq.read_table(path).to_pylist():
                preds = r["predictions"] if isinstance(r["predictions"], list) else [r["predictions"]]
                prompt = json.dumps(r["full_prompt"], ensure_ascii=False, sort_keys=True)
                yield {"run": run, "task": task, "prompt_sha": hashlib.sha256(prompt.encode()).hexdigest()[:16],
                       "samples": len(preds),
                       # An answer that is still inside its thinking when it stops starts with the thinking tag.
                       "unfinished": round(mean(p.startswith("<think") for p in preds), 4),
                       "score": round(score_of(r["metrics"], task), 4),
                       "answer_chars": round(mean(len(p) for p in preds))}


def tau2_rows(raw):
    for path in sorted(glob.glob(os.path.join(raw, "tau2", "*", "results.json"))):
        run = os.path.basename(os.path.dirname(path))
        for s in json.load(open(path))["simulations"]:
            messages = s.get("messages") or []
            calls = [c for m in messages if m.get("role") == "assistant" for c in (m.get("tool_calls") or [])]
            # tau2 answers a call to a tool the caller does not have with "Error: Tool '<name>' not found."
            not_found = [str(m.get("content")).split("'")[1] for m in messages if m.get("role") == "tool"
                         and m.get("requestor") == "assistant" and str(m.get("content")).startswith("Error: Tool '")]
            reward = (s.get("reward_info") or {}).get("reward")
            yield {"run": run, "task_id": s["task_id"], "trial": s.get("trial"),
                   "reward": "" if reward is None else reward, "termination": s.get("termination_reason"),
                   "duration_s": round(s.get("duration") or 0), "messages": len(messages),
                   "agent_tool_calls": len(calls),
                   "tool_errors": sum(1 for m in messages if m.get("role") == "tool" and m.get("error")),
                   "missing_tool_calls": len(not_found), "missing_tools": ";".join(sorted(set(not_found)))}


def write(path, rows):
    rows = list(rows)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(path, len(rows), "rows")


raw, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
write(os.path.join(out, "swallow.csv"), swallow_rows(raw))
write(os.path.join(out, "tau2.csv"), tau2_rows(raw))
