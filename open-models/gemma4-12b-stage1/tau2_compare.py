# Summarize each tau2 run, then compare two runs on the tasks both scored: passes, tasks passed by only one run with
# an exact two-sided sign test, how the conversations ended and how often the agent called tools it does not have.
# Arguments: scores/tau2.csv, then two run specs such as tv and tb+tb_retry; in a spec joined with "+", a later run
# replaces an earlier one's result for the same task.
import collections, csv, math, statistics, sys

by_run = collections.defaultdict(list)
for r in csv.DictReader(open(sys.argv[1])):
    by_run[r["run"]].append(r)


def scored(r):
    return r["reward"] != "" and r["termination"] != "infrastructure_error"


def passed(r):
    return float(r["reward"]) >= 1.0


for run, sims in sorted(by_run.items()):
    done = [s for s in sims if scored(s)]
    print(f"== {run}: conversations {len(sims)}, scored {len(done)}, passed {sum(map(passed, done))}")
    print("   endings", dict(collections.Counter(s["termination"] for s in sims)))
    print(f"   median seconds {statistics.median(int(s['duration_s']) for s in sims):.0f}, "
          f"longest {max(int(s['duration_s']) for s in sims)}")
    missing = collections.Counter(t for s in sims for t in s["missing_tools"].split(";") if t)
    print("   conversations calling each missing tool:", dict(missing.most_common(8)))


def load(spec):
    tasks = {}
    for run in spec.split("+"):
        tasks.update({s["task_id"]: s for s in by_run[run] if scored(s)})
    return tasks


if len(sys.argv) > 3:
    names = sys.argv[2:4]
    a, b = (load(name) for name in names)
    common = sorted(set(a) & set(b))
    n = len(common)
    print(f"== {names[0]} vs {names[1]}: scored {len(a)} and {len(b)}, both {n}")
    for name, run in zip(names, (a, b)):
        rows = [run[t] for t in common]
        wins = sum(map(passed, rows))
        print(f"   {name} passed {wins}/{n} ({wins / n:.1%}), endings",
              dict(collections.Counter(r["termination"] for r in rows)))
        print(f"   {name} per conversation: agent tool calls {statistics.fmean(int(r['agent_tool_calls']) for r in rows):.1f}, "
              f"calls to missing tools {statistics.fmean(int(r['missing_tool_calls']) for r in rows):.1f}; "
              f"conversations with such a call {sum(int(r['missing_tool_calls']) > 0 for r in rows)}")
    only_a = sum(passed(a[t]) and not passed(b[t]) for t in common)
    only_b = sum(passed(b[t]) and not passed(a[t]) for t in common)
    k, m = min(only_a, only_b), only_a + only_b
    p = min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m) if m else 1.0
    print(f"   passed by only one: {names[0]} {only_a}, {names[1]} {only_b}; two-sided sign test p = {p:.2f}")
