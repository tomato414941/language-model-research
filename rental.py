"""Rents a GPU on Vast.ai, an H100 unless another is named, sends it this repository and the data, starts
a script, fetches what it made and removes the machine.

Commands: offers [GPU], rent [GPU], send [HOST:DIRECTORY [PATH...]], start SCRIPT [ARGUMENT...], shell COMMAND,
fetch, destroy, guard MINUTES. `send` sends the committed code, and the data if given the machine and directory
that hold it: prepare.py's climbmix directory, harness.py's questions.jsonl, the harness's datasets cache and
constraints.txt, the versions that a try without a GPU ran with, unless other paths are named. `guard`, left
running in the background, removes the machine after that long if nothing else has.

It rents through runs/vast_rental.py of the language-model repository, checked out next to this one,
so that both projects follow one rental policy and share one record of the machines that failed. That
module delivers the Vast.ai key. Nothing here prints an address or a key.
"""

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parent
sys.path.insert(0, str(REPOSITORY.parent / "language-model" / "runs"))
import vast_rental as rental  # noqa: E402

HERE = REPOSITORY / "out" / "rental"
STATE = HERE / "state.json"
REMOTE = "/root/lmr"
# The image of the baseline, so that a change of PyTorch does not mix into a comparison with it.
IMAGE = "pytorch/pytorch:2.15.0-cuda13.2-cudnn9-runtime"
DISK_GB = 60
MAX_DOLLARS_PER_HOUR = 2.80
TRIES = 2
LABEL = "language-model-research"
# What a comparison's costs are measured on. Work outside one, like scoring saved weights, can name a cheaper GPU.
GPU = "H100 SXM"
DATA = ("climbmix", "questions.jsonl", "datasets", "constraints.txt")
QUERY = {
    "num_gpus": {"eq": 1},
    "cuda_max_good": {"gte": 13.2},
    "reliability2": {"gte": 0.97},
    "inet_down": {"gte": 500},
    "disk_space": {"gte": DISK_GB},
    "cpu_ram": {"gte": 32000},
    "cpu_cores_effective": {"gte": 4},
}
OFFER_FIELDS = ("id", "machine_id", "dph_total", "inet_up", "inet_down", "reliability2", "gpu_mem_bw", "cuda_max_good")


def instance():
    return json.loads(STATE.read_text())["instance"]


def find_offers(gpu=GPU):
    query = {**QUERY, "gpu_name": {"eq": gpu}}
    offers = [offer for offer in rental.search(query, DISK_GB) if offer["dph_total"] <= MAX_DOLLARS_PER_HOUR]
    return sorted(offers, key=lambda offer: (offer["dph_total"], -offer.get("inet_up", 0)))


def show_offers(gpu=GPU):
    print(json.dumps([{key: offer.get(key) for key in OFFER_FIELDS} for offer in find_offers(gpu)][:5]))


def rent(gpu=GPU):
    offers = find_offers(gpu)
    if not offers:
        raise SystemExit("no offer within the price")
    began = time.time()
    rented = rental.rent(offers, HERE / "diagnostics", TRIES, image=IMAGE, disk=DISK_GB, label=LABEL)
    if rented is None:
        raise SystemExit("no offer started")
    number, offer = rented
    HERE.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"instance": number, "dph": offer["dph_total"], "machine": offer["machine_id"]}))
    print(json.dumps({"rented": number, "dph": offer["dph_total"], "machine": offer["machine_id"],
                      "where": offer.get("geolocation"), "seconds_to_ready": round(time.time() - began)}))


def ssh_to_instance():
    """The ssh command for the instance, without its last argument, the command to run there. Like rental.login,
    it tries the host's own port and then Vast.ai's proxy, since some hosts' own port cannot be reached."""
    row = rental.vast().show_instance(instance())
    mapped = (row.get("ports") or {}).get("22/tcp") or []
    routes = [(row["public_ipaddr"], mapped[0]["HostPort"])] if mapped else []
    routes.append((row["ssh_host"], row["ssh_port"]))
    for host, port in routes:
        command = ["ssh", "-i", str(rental.SSH_KEY), "-p", str(port), "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
                   "-o", "StrictHostKeyChecking=accept-new", "-o", f"UserKnownHostsFile={rental.KNOWN_HOSTS / str(instance())}",
                   "-o", "LogLevel=ERROR", f"root@{host}"]
        # 255 is ssh's own failure; anything else means the connection was made.
        if subprocess.run([*command, "true"], capture_output=True).returncode != 255:
            break
    return command


def stream(source, target_command):
    """Pipe a local command's output into a command on the instance, and report how long it took."""
    began = time.time()
    producer = subprocess.Popen(source, stdout=subprocess.PIPE)
    done = subprocess.run([*ssh_to_instance(), target_command], stdin=producer.stdout, capture_output=True, text=True)
    producer.stdout.close()
    producer.wait()
    print(json.dumps({"seconds": round(time.time() - began), "source": producer.returncode, "target": done.returncode,
                      "error": rental.hide_addresses(done.stderr)[-500:]}))
    if producer.returncode or done.returncode:
        raise SystemExit("sending failed")


def send(data_source=None, *paths):
    stream(["git", "-C", str(REPOSITORY), "archive", "HEAD"], f"mkdir -p {REMOTE} && tar -x -C {REMOTE}")
    if data_source is None:
        return
    host, directory = data_source.split(":", 1)
    names = " ".join(shlex.quote(path) for path in paths or DATA)
    remote_tar = f"tar -C {shlex.quote(directory)} -cf - {names}"
    stream(["ssh", "-o", "BatchMode=yes", host, remote_tar], f"mkdir -p {REMOTE}/data && tar -x -C {REMOTE}/data")


def shell(command, timeout=120):
    completed = rental.login(instance(), command, timeout=int(timeout))
    print(completed.stdout[-6000:], end="")
    if completed.returncode:
        print(rental.hide_addresses(completed.stderr)[-1500:], file=sys.stderr)
    return completed.returncode


def start(script, *arguments):
    words = " ".join(shlex.quote(word) for word in [script, *arguments])
    # The subshell lets go of the connection at once; otherwise ssh waits for the script to end.
    return shell(f"cd {REMOTE} && (nohup bash {words} > run.log 2>&1 < /dev/null &) ; echo started")


def fetch():
    HERE.mkdir(parents=True, exist_ok=True)
    began = time.time()
    source = subprocess.Popen([*ssh_to_instance(), f"tar -C {REMOTE} -cf - run.log results models"], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=False)
    unpack = subprocess.run(["tar", "-x", "-C", str(HERE)], stdin=source.stdout, capture_output=True, text=True)
    source.stdout.close()
    error = source.stderr.read().decode(errors="replace")
    source.wait()
    print(json.dumps({"seconds": round(time.time() - began), "source": source.returncode, "unpack": unpack.returncode,
                      "error": rental.hide_addresses(error + unpack.stderr)[-400:]}))
    return source.returncode or unpack.returncode


def destroy():
    number = instance()
    answer = rental.vast().destroy_instance(number)
    STATE.unlink()
    left = [row["id"] for row in rental.vast().show_instances() if row.get("label") == LABEL and row["id"] != number]
    print(json.dumps({"destroyed": number, "answer": answer, "other_instances_of_this_project": left}, default=str)[:300])


def guard(minutes):
    number = instance()
    time.sleep(minutes * 60)
    # The machine may be gone by then, and one rented since is not the one this was left to watch.
    if STATE.exists() and instance() == number:
        destroy()


def main():
    command, *arguments = sys.argv[1:] or ["help"]
    if command == "help":
        print(__doc__)
        return 0
    if "VAST_API_KEY" not in os.environ:
        limit = int(arguments[0]) * 60 + 300 if command == "guard" else 3000
        done = rental.with_key(Path(__file__).resolve(), command, *arguments, timeout=limit)
        print(done.stdout, end="")
        print(rental.hide_addresses(done.stderr), end="", file=sys.stderr)
        return done.returncode
    actions = {"offers": show_offers, "rent": rent, "send": send, "start": start, "shell": shell, "fetch": fetch,
               "destroy": destroy, "guard": lambda minutes: guard(int(minutes))}
    return actions[command](*arguments) or 0


if __name__ == "__main__":
    sys.exit(main())
