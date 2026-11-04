"""Rents the H-Net comparison's machine through rental.py, with its own record of the machine and its own label, so
that it never touches a bundle's rental.

Commands are rental.py's (offers [GPU], rent [GPU], send, start SCRIPT [ARGUMENT...], shell COMMAND, fetch, destroy,
guard MINUTES), whose record and fetched files go to out/h-net-rental, and `sources`, which sends the file that
mint_sources.py wrote there through ssh's input, so that no process list or log shows its links, and then removes it.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rental as research  # noqa: E402

research.HERE = research.REPOSITORY / "out" / "h-net-rental"
research.STATE = research.HERE / "state.json"
research.LABEL = "language-model-research-h-net"
SOURCES = research.HERE / "sources.json"


def send_sources():
    with SOURCES.open("rb") as source:
        done = subprocess.run([*research.ssh_to_instance(), f"umask 077 && cat > {research.REMOTE}/hnet-sources.json"],
                              stdin=source, capture_output=True)
    print(json.dumps({"sent": done.returncode == 0}))
    if done.returncode == 0:
        SOURCES.unlink()
    return done.returncode


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
               "guard": lambda minutes: research.guard(int(minutes)), "sources": send_sources}
    return actions[command](*arguments) or 0


if __name__ == "__main__":
    sys.exit(main())
