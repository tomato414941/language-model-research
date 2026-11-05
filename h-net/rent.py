"""Rents the H-Net comparison's machine through rental.py, with its own record of the machine and its own label, so
that it never touches a bundle's rental.

Commands are rental.py's: offers [GPU], rent [GPU], send [HOST:DIRECTORY [PATH...]], start SCRIPT [ARGUMENT...],
shell COMMAND, fetch, destroy, guard MINUTES; send-data HOST:DIRECTORY DATASET..., which sends each dataset's
meta.pkl, tokenizer.json and val.bin and then its train.bin, which takes its name only once all of it has come, so
that run.sh, started while it is sent, can wait for it; send-olmes HOST:DIRECTORY, which sends frame.py's requests
(olmes-requests) and the Hugging Face cache of its tasks' data (hf) from that data directory and then marks them as
come with data/olmes.sent; and send-commit COMMIT DIRECTORY, which sends that commit's code to DIRECTORY inside the
sent repository: "." sends the code of a commit other than HEAD, and a directory an earlier version to be timed on
the same machine. The record and the fetched files go to out/h-net-rental.
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


def send_data(source, *datasets):
    host, directory = source.split(":", 1)
    for dataset in datasets:
        here = f"{shlex.quote(directory)}/{shlex.quote(dataset)}"
        there = f"{research.REMOTE}/data/{shlex.quote(dataset)}"
        small = f"tar -C {here} -cf - meta.pkl tokenizer.json val.bin"
        research.stream(["ssh", "-o", "BatchMode=yes", host, small], f"mkdir -p {there} && tar -x -C {there}")
        research.stream(["ssh", "-o", "BatchMode=yes", host, f"cat {here}/train.bin"], f"cat > {there}/train.bin.part")
        if research.shell(f"mv {there}/train.bin.part {there}/train.bin"):
            raise SystemExit("naming train.bin failed")


def send_olmes(source):
    research.send_data(source, "olmes-requests", "hf")
    if research.shell(f"touch {research.REMOTE}/data/olmes.sent"):
        raise SystemExit("marking olmes's data failed")


def send_commit(commit, directory):
    there = f"{research.REMOTE}/{shlex.quote(directory)}"
    research.stream(["git", "-C", str(research.REPOSITORY), "archive", commit], f"mkdir -p {there} && tar -x -C {there}")


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
               "guard": lambda minutes: research.guard(int(minutes)), "send-data": send_data,
               "send-olmes": send_olmes, "send-commit": send_commit}
    return actions[command](*arguments) or 0


if __name__ == "__main__":
    sys.exit(main())
