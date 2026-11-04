"""Rents the H-Net comparison's machine through rental.py, with its own record of the machine and its own label, so
that it never touches a bundle's rental.

Commands are rental.py's: offers [GPU], rent [GPU], send [HOST:DIRECTORY [PATH...]], start SCRIPT [ARGUMENT...],
shell COMMAND, fetch, destroy, guard MINUTES. The record and the fetched files go to out/h-net-rental.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rental as research  # noqa: E402

research.HERE = research.REPOSITORY / "out" / "h-net-rental"
research.STATE = research.HERE / "state.json"
research.LABEL = "language-model-research-h-net"


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
               "guard": lambda minutes: research.guard(int(minutes))}
    return actions[command](*arguments) or 0


if __name__ == "__main__":
    sys.exit(main())
