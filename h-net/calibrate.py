"""Measures on this machine what an update of each model in the H-Net comparison costs, and how many updates the
baseline's cost buys each, as speed.py does for the bundles. The baseline is timed here too, so that the machine's
own speed cancels out.

H-Net is timed with a chunk at every 6 bytes, the ratio loss's target: early in training its routing begins a chunk at
about every other byte, which a short timing would take for the cost of the whole run.

Writes results/hnet-speed.json with each model's costs and updates. Usage: python h-net/calibrate.py [NAME...]
"""

import json
import sys

sys.path.insert(0, ".")
from speed import RESULTS, SLOW_BASELINE_SECONDS, config_value, measure, updates_for_the_cost  # noqa: E402

COST = {"name": "hnet-cost", "config": "h-net/configs/cost.py", "attention": "sdpa", "options": {}}
TARGET_BYTES_PER_CHUNK = 6
MODELS = {
    "hnet-bpe": "h-net/configs/bpe.py",
    "hnet-mamba2": "h-net/configs/hnet_mamba2.py",
    "hnet-mamba3": "h-net/configs/hnet_mamba3.py",
    "hnet-gated-deltanet": "h-net/configs/hnet_gated_deltanet.py",
}


def way(name):
    options = {"stream_bfloat16": True}
    if name != "hnet-bpe":
        options["fixed_chunks"] = TARGET_BYTES_PER_CHUNK
    return {"name": name, "config": MODELS[name], "attention": "fa2", "options": options}


def main():
    names = sys.argv[1:] or list(MODELS)
    RESULTS.mkdir(exist_ok=True)
    costs = {"baseline": measure(COST, [])}
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
