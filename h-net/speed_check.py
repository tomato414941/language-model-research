"""Times H-Net's training updates on this machine without research's data: train.py as it is, run in this process
on random bytes, with a hook after each optimizer step that waits for the GPU and notes the time and the memory, and
that can profile some of the updates. The bytes' values do not change the cost when the config begins a chunk at
every fixed number of bytes (fixed_chunks).

`data` writes the random bytes as bytes_from_tokens.py writes a dataset, in documents of about FineWeb's length;
`tokens` writes random tokens of research's BPE vocabulary the same way, so that a bundle can be timed beside H-Net.
`run` writes a summary to OUTPUT: the time of each update, the median after the first ones, which compile, the peak
memory and the allocator's retries; with --profile=FIRST:LAST also the updates after FIRST up to LAST, profiled, as
a table of what took the GPU's time and as a trace beside OUTPUT, and left out of the median. --unfused runs Mamba-2
on its unfused path, with PyTorch's convolution, as H-Net ran before causal_conv.py.

Usage:
  python h-net/speed_check.py data DIRECTORY
  python h-net/speed_check.py tokens DIRECTORY
  python h-net/speed_check.py run OUTPUT [--profile=FIRST:LAST] [--unfused] CONFIG [TRAIN_OPTION...]
"""

import json
import pickle
import runpy
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

SIZE = 1 << 27  # bytes or tokens to train on
BLOCK_SIZE = 3840  # what bytes_from_tokens.py gives FineWeb's rows
DOCUMENT_BYTES = 3100  # about FineWeb's mean
BYTES_PER_TOKEN = 3.81  # FineWeb's in research's tokenizer
VOCABULARY = 8192  # prepare.py's
SETTLING_UPDATES = 5


def write_data(directory, vocabulary=None):
    """Bytes with H-Net's BOS (256) before each document, or tokens of `vocabulary` with its last as the BOS."""
    directory = Path(directory)
    directory.mkdir(parents=True)
    bos = 256 if vocabulary is None else vocabulary - 1
    document = DOCUMENT_BYTES if vocabulary is None else round(DOCUMENT_BYTES / BYTES_PER_TOKEN)
    generator = np.random.default_rng(0)
    for split, size in (("train", SIZE), ("val", SIZE // 16)):
        values = generator.integers(0, bos, size, dtype=np.uint16)
        heads = np.cumsum(generator.integers(1, 2 * document, size // document + 1))
        values[heads[heads < size]] = bos
        values[0] = bos
        values.tofile(directory / f"{split}.bin")
    with (directory / "meta.pkl").open("wb") as meta:
        pickle.dump({"vocab_size": bos + 1}, meta)
    (directory / "tokenizer.json").write_text(json.dumps({"added_tokens": [{"id": bos, "content": "<|bos|>"}]}) + "\n")
    if vocabulary is None:
        (directory / "rows.json").write_text(json.dumps({"block_size": BLOCK_SIZE, "source": "random bytes"}) + "\n")


def use_unfused_mamba2():
    sys.path[:0] = [".", "h-net"]
    from hnet_model import without_package_imports

    without_package_imports("mamba_ssm")
    from mamba_ssm.modules import mamba2

    original = mamba2.Mamba2.__init__
    mamba2.Mamba2.__init__ = lambda self, *args, **kwargs: original(self, *args, **{**kwargs, "use_mem_eff_path": False})


class Watch:
    """Called after every optimizer step."""

    def __init__(self, output, profile):
        self.output = Path(output)
        self.profile = profile
        self.ends = []
        self.profiler = None
        self.table = None
        self.busy = None

    def __call__(self, optimizer, args, kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.ends.append(time.perf_counter())
        update = len(self.ends)
        if self.profile and update == self.profile[0]:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            self.profiler = torch.profiler.profile(activities=activities)
            self.profiler.start()
            self.began = time.perf_counter()
        elif self.profile and update == self.profile[1]:
            wall = time.perf_counter() - self.began
            self.profiler.stop()
            averages = self.profiler.key_averages()
            self.table = averages.table(sort_by="self_device_time_total", row_limit=60, max_name_column_width=90)
            kernels = sum(event.self_device_time_total for event in averages) / 1e6
            self.busy = {"wall_seconds": wall, "gpu_seconds": kernels, "updates": self.profile[1] - self.profile[0]}
            self.profiler.export_chrome_trace(str(self.output.with_suffix(".trace.json.gz")))
        self.write()

    def write(self):
        # times[i] is update i + 2's, and the profiled ones begin with update profile[0] + 1.
        times = [later - earlier for earlier, later in zip(self.ends, self.ends[1:])]
        measured = times[SETTLING_UPDATES : self.profile[0] - 1 if self.profile else None]
        summary = {
            "seconds_per_update": statistics.median(measured) if measured else None,
            "update_seconds": [round(seconds, 4) for seconds in times],
            "profiled": self.busy,
        }
        if torch.cuda.is_available():
            summary.update(peak_allocated_gb=torch.cuda.max_memory_allocated() / 2**30,
                           peak_reserved_gb=torch.cuda.max_memory_reserved() / 2**30,
                           alloc_retries=torch.cuda.memory_stats().get("num_alloc_retries", 0))
        self.output.write_text(json.dumps(summary, indent=1) + "\n")
        if self.table is not None:
            self.output.with_suffix(".profile.txt").write_text(self.table + "\n")


def run(output, arguments):
    profile = None
    while arguments and (arguments[0] == "--unfused" or arguments[0].startswith("--profile=")):
        option = arguments.pop(0)
        if option == "--unfused":
            use_unfused_mamba2()
        else:
            profile = tuple(int(update) for update in option.split("=", 1)[1].split(":"))
    from torch.optim.optimizer import register_optimizer_step_post_hook

    register_optimizer_step_post_hook(Watch(output, profile))
    # As `python train.py` would: the repository's root first on the path, and train.py's arguments.
    sys.path.insert(0, str(Path.cwd()))
    sys.argv = ["train.py", *arguments]
    runpy.run_path("train.py", run_name="__main__")


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("data", "tokens", "run"):
        raise SystemExit(__doc__)
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3:])
    else:
        write_data(sys.argv[2], VOCABULARY if sys.argv[1] == "tokens" else None)


if __name__ == "__main__":
    main()
