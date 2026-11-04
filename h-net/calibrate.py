"""Measures on this machine what an update of each H-Net costs, and how many updates the baseline's cost buys each, as
speed.py does for the bundles. The baseline is timed here too, on the BPE dataset that it would read, so that the
machine's own speed cancels out.

H-Net is timed with a chunk at every 6 bytes, the ratio loss's target: early in training its routing begins a chunk at
about every other byte, which a short timing would take for the cost of the whole run.

Writes results/hnet-speed.json with each model's costs and updates.
Usage: python h-net/calibrate.py BPE_DATASET [NAME...]
"""

import json
import sys

sys.path.insert(0, ".")
from speed import RESULTS, SLOW_BASELINE_SECONDS, config_value, measure, updates_for_the_cost  # noqa: E402

TARGET_BYTES_PER_CHUNK = 6
MODELS = {
    "hnet-mamba2": "h-net/configs/hnet_mamba2.py",
    "hnet-mamba3": "h-net/configs/hnet_mamba3.py",
    "hnet-gated-deltanet": "h-net/configs/hnet_gated_deltanet.py",
}


def way(name):
    return {"name": name, "config": MODELS[name], "attention": "fa2",
            "options": {"stream_bfloat16": True, "fixed_chunks": TARGET_BYTES_PER_CHUNK}}


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    dataset, names = sys.argv[1], sys.argv[2:] or list(MODELS)
    RESULTS.mkdir(exist_ok=True)
    baseline = {"name": "hnet-baseline", "config": "configs/baseline.py", "attention": "sdpa",
                "options": {"dataset": dataset}}
    costs = {"baseline": measure(baseline, [])}
    print(f"baseline: {costs['baseline']}", file=sys.stderr, flush=True)
    if costs["baseline"] is None:
        raise SystemExit("The baseline did not run, so nothing can be compared.")
    if costs["baseline"]["seconds_per_update"] > SLOW_BASELINE_SECONDS:
        raise SystemExit(f"This machine is too slow: {costs['baseline']}")
    baseline_updates = config_value("configs/baseline.py", "max_iters")
    updates = {}
    for name in names:
        costs[name] = measure(way(name), [])
        print(f"{name}: {costs[name]}", file=sys.stderr, flush=True)
        if costs[name] is not None:
            updates[name] = updates_for_the_cost(baseline_updates, costs["baseline"], costs[name])
    (RESULTS / "hnet-speed.json").write_text(json.dumps({"costs": costs, "updates": updates}, indent=1) + "\n")
    print(json.dumps(updates))


if __name__ == "__main__":
    main()
