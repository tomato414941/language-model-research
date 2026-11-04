"""Rents the H-Net comparison's machine through rental.py, with its own record of the machine and its own label, so
that it never touches a bundle's rental.

Commands are rental.py's: offers [GPU], rent [GPU], send [HOST:DIRECTORY [PATH...]], start SCRIPT [ARGUMENT...],
shell COMMAND, fetch, destroy, guard MINUTES; and send-head HOST:DIRECTORY BYTES DATASET..., which sends each
dataset's meta.pkl, tokenizer.json and val.bin and only the first BYTES of its train.bin, whose documents are already
in a random order, so that a run that reads less than a dataset waits less for it. The record and the fetched files
go to out/h-net-rental.
"""

import os
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rental as research  # noqa: E402

research.HERE = research.REPOSITORY / "out" / "h-net-rental"
research.STATE = research.HERE / "state.json"
research.LABEL = "language-model-research-h-net"


def send_head(source, size, *datasets):
    host, directory = source.split(":", 1)
    for dataset in datasets:
        here = f"{shlex.quote(directory)}/{shlex.quote(dataset)}"
        there = f"{research.REMOTE}/data/{shlex.quote(dataset)}"
        small = f"tar -C {here} -cf - meta.pkl tokenizer.json val.bin"
        research.stream(["ssh", "-o", "BatchMode=yes", host, small], f"mkdir -p {there} && tar -x -C {there}")
        head = f"head -c {int(size)} {here}/train.bin"
        research.stream(["ssh", "-o", "BatchMode=yes", host, head], f"cat > {there}/train.bin")


def main():
    command, *arguments = sys.argv[1:] or ["help"]
    if command == "help":
        print(__doc__)
        return 0
    if "VAST_API_KEY" not in os.environ:
        limit = int(arguments[0]) * 60 + 300 if command == "guard" else 3000
        done = research.rental.with_key(Path(__file__).resolve(), command, *arguments, timeout=limit)
        print(done.stdout, end="")
        print(research.rental.hide_addresses(done.stderr), end="", file=sys.stderr)
        return done.returncode
    actions = {"offers": research.show_offers, "rent": research.rent, "send": research.send, "start": research.start,
               "shell": research.shell, "fetch": research.fetch, "destroy": research.destroy,
               "guard": lambda minutes: research.guard(int(minutes)), "send-head": send_head}
    return actions[command](*arguments) or 0


if __name__ == "__main__":
    sys.exit(main())
