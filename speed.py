"""Measures what each way of running the bundle costs on this machine, and prints the training
options of the cheapest. Every way reads the same tokens per update, so a cheaper one buys more updates
for the cost of the baseline, which is run here too so that the machine's own speed cancels out.

It first compares the ways of running attention, then tries the type of the values through the layers and
the number of sequences per pass with the best so far, since trying every combination would cost several
times as much. It also times the bundle without each part whose cost is in doubt. Those runs change the model,
so they are only recorded and never chosen. The times go to results/speed.json and the runs' logs beside it.

With --also=CONFIG, it then times that configuration in the way it chose, and prints a second line of options,
the updates that buy that configuration the same cost.

Usage: speed.py [--also=CONFIG] [TRAIN_OPTION...]. The options go to every run, so that a small model can try
this script without a GPU.
"""

import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

RESULTS = Path("results")
MEASURED_UPDATES = 25
# The first updates include compiling and the settling of the GPU's caches.
SETTLING_UPDATES = 5
RUN_SECONDS_LIMIT = 600
# The baseline took about 0.83 seconds per update on its H100, and some hosts run at half the usual speed.
SLOW_BASELINE_SECONDS = 0.83 * 1.5
# The first bundle measured these two as fast as the other ways, which take extra installing.
ATTENTIONS = ("sdpa", "fa2")
USUAL_PASS = (128, 4)
OTHER_PASSES = ((64, 8), (256, 2))
BUNDLE_CONFIG = "configs/bundle3.py"
PARTS_IN_DOUBT = {
    "without-count": {"count_table": ""},
    "without-dispersion": {"dispersion": 0.0},
}


def bundle_way(attention, batch_size=USUAL_PASS[0], accumulation=USUAL_PASS[1], block_size=1024, compile_model=True,
               stream=False, part=None):
    """A way of running the bundle. All of them read 524,288 tokens per update."""
    name = f"{attention}-{batch_size}x{accumulation}"
    options = {
        "attention": attention,
        "batch_size": batch_size,
        "gradient_accumulation_steps": accumulation,
        "block_size": block_size,
        "compile": compile_model,
    }
    if block_size != 1024:
        name += f"-context-{block_size}"
    if not compile_model:
        name += "-uncompiled"
    if stream:
        name += "-stream"
        options["stream_bfloat16"] = True
    if part is not None:
        name += f"-{part}"
        options.update(PARTS_IN_DOUBT[part])
    return {
        "name": name,
        "config": BUNDLE_CONFIG,
        "attention": attention,
        "keeps_model": block_size == 1024 and part is None,
        "options": options,
    }


BASELINE = {"name": "baseline", "config": "configs/baseline.py", "attention": "sdpa", "keeps_model": False, "options": {}}


def as_options(options):
    return [f"--{key}={value}" for key, value in options.items()]


def seconds_per_update(log):
    """The median time of the updates after the settling ones, or None with too few of them."""
    times = [float(seconds) / 1000 for seconds in re.findall(r"^iter \d+: .*time ([\d.]+)ms", log, re.MULTILINE)]
    measured = times[SETTLING_UPDATES:]
    return statistics.median(measured) if len(measured) >= SETTLING_UPDATES else None


def attention_used(log):
    found = re.search(r"^attention: (\S+)", log, re.MULTILINE)
    return found.group(1) if found else None


def stream_in_bfloat16(log):
    found = re.search(r"^stream: (.+)", log, re.MULTILINE)
    return found is not None and "bfloat16" in found.group(1)


def cost(log, total_seconds, updates, attention, stream=False):
    """Seconds per update and the seconds around the updates, or None if the run did not do what was asked."""
    seconds = seconds_per_update(log)
    if seconds is None or attention_used(log) != attention or stream_in_bfloat16(log) != stream:
        return None
    return {"seconds_per_update": seconds, "startup_seconds": max(total_seconds - updates * seconds, 0.0)}


def measure(way, extra):
    """What the way costs, or None if it failed or fell back to another way of running attention."""
    # A run this short reaches none of the updates whose weights the bundle averages at the end of its training.
    command = [sys.executable, "-u", "train.py", way["config"], *as_options(way["options"]), f"--out_dir=out/speed/{way['name']}",
               f"--max_iters={MEASURED_UPDATES}", "--eval_interval=1000000", "--eval_iters=1", "--log_interval=1",
               "--average_last=0", *extra]
    began = time.monotonic()
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=RUN_SECONDS_LIMIT)
        log, failed = done.stdout + done.stderr, done.returncode != 0
    except subprocess.TimeoutExpired as expired:
        log, failed = f"{expired.stdout}\n{expired.stderr}\nThe run took too long.", True
    total_seconds = time.monotonic() - began
    (RESULTS / f"speed-{way['name']}.log").write_text(log)
    # The loop stops after one update more than max_iters.
    stream = way["options"].get("stream_bfloat16", False)
    return None if failed else cost(log, total_seconds, MEASURED_UPDATES + 1, way["attention"], stream)


def agrees_with_default(way):
    """Whether this way of running gives what PyTorch's default gives, on this machine."""
    if way == "sdpa":
        return True
    done = subprocess.run([sys.executable, "attention_check.py", way], capture_output=True, text=True)
    (RESULTS / f"speed-check-{way}.log").write_text(done.stdout + done.stderr)
    return done.returncode == 0


def updates_for_the_cost(baseline_updates, baseline, way):
    """How many updates of the way take the time of the baseline's updates, counting the time around them."""
    budget = baseline_updates * baseline["seconds_per_update"] + baseline["startup_seconds"]
    return round((budget - way["startup_seconds"]) / way["seconds_per_update"])


def cheapest(ways, costs, baseline_updates):
    """The way that buys the most updates, among those that worked and keep the bundle's model."""
    usable = [way for way in ways if way["keeps_model"] and costs.get(way["name"]) is not None]
    if not usable:
        return None
    return max(usable, key=lambda way: updates_for_the_cost(baseline_updates, costs["baseline"], costs[way["name"]]))


def same_way(way, config):
    """The way of running, for another configuration."""
    return {**way, "name": f"{way['name']}-{Path(config).stem}", "config": config}


def training_options(way, updates):
    options = {key: value for key, value in way["options"].items() if key != "block_size"}
    return as_options({**options, "max_iters": updates, "lr_decay_iters": updates, "eval_interval": updates})


def config_value(path, name):
    values = {}
    exec(Path(path).read_text(), values)
    return values[name]


def main():
    also = [argument.split("=", 1)[1] for argument in sys.argv[1:] if argument.startswith("--also=")]
    extra = [argument for argument in sys.argv[1:] if not argument.startswith("--also=")]
    RESULTS.mkdir(exist_ok=True)
    baseline_updates = config_value("configs/baseline.py", "max_iters")
    costs = {}

    def run(way):
        costs[way["name"]] = measure(way, extra)
        print(f"{way['name']}: {costs[way['name']]}", file=sys.stderr, flush=True)

    run(BASELINE)
    if costs["baseline"] is None:
        raise SystemExit("The baseline did not run, so nothing can be compared.")
    if costs["baseline"]["seconds_per_update"] > SLOW_BASELINE_SECONDS:
        raise SystemExit(f"This machine is too slow: {costs['baseline']}")

    agreeing = {way: agrees_with_default(way) for way in (*ATTENTIONS, "stream")}
    print(f"agrees with PyTorch's default: {agreeing}", file=sys.stderr, flush=True)
    ways = [bundle_way(attention) for attention in ATTENTIONS if agreeing[attention]]
    for way in ways:
        run(way)
    best = cheapest(ways, costs, baseline_updates)
    if best is None:
        raise SystemExit("No way of running the bundle worked.")

    if agreeing["stream"]:
        ways.append(bundle_way(best["attention"], stream=True))
        run(ways[-1])
    best = cheapest(ways, costs, baseline_updates)
    stream = "stream_bfloat16" in best["options"]

    for batch_size, accumulation in OTHER_PASSES:
        ways.append(bundle_way(best["attention"], batch_size, accumulation, stream=stream))
        run(ways[-1])

    chosen = cheapest(ways, costs, baseline_updates)
    options = chosen["options"]
    for part in PARTS_IN_DOUBT:
        run(bundle_way(chosen["attention"], options["batch_size"], options["gradient_accumulation_steps"],
                       stream="stream_bfloat16" in options, part=part))
    updates = updates_for_the_cost(baseline_updates, costs["baseline"], costs[chosen["name"]])
    lines = [" ".join(training_options(chosen, updates))]
    also_updates = {}
    for config in also:
        way = same_way(chosen, config)
        run(way)
        if costs[way["name"]] is None:
            raise SystemExit(f"{config} did not run in the chosen way.")
        also_updates[config] = updates_for_the_cost(baseline_updates, costs["baseline"], costs[way["name"]])
        lines.append(" ".join(training_options(way, also_updates[config])))
    (RESULTS / "speed.json").write_text(json.dumps(
        {"agrees_with_default": agreeing, "costs": costs, "chosen": chosen["name"], "updates": updates,
         "also_updates": also_updates}, indent=1) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
