"""Checks on the rented GPU before anything costs much: that each kind of outer layer runs forward and backward in
bfloat16 with finite values and reads only the bytes before each position, and that packed attention gives what
PyTorch's attention gives row by row. Prints a line for each, and fails only if packed attention does, which every
H-Net model needs; an outer layer that fails here fails its timing too, and its model is skipped.

Usage: python h-net/check_gpu.py [KIND...], with KIND among mamba2, mamba3 and gated_deltanet, all of them if none
"""

import json
import sys
import time

import torch
import torch.nn.functional as F

sys.path[:0] = [".", "h-net"]
from hnet_model import OuterLayer, causal_within_rows  # noqa: E402

KINDS = ("mamba2", "mamba3", "gated_deltanet")


def check_outer(kind):
    torch.manual_seed(0)
    layer = OuterLayer(kind, 512, 0).cuda()
    x = torch.randn(4, 3328, 512, device="cuda", requires_grad=True)
    torch.cuda.synchronize()
    began = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = layer(x)
    y.float().square().mean().backward()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - began
    changed = x.detach().clone()
    changed[:, 2000:] = torch.randn_like(changed[:, 2000:])
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        before, after = layer(x.detach())[:, :2000], layer(changed)[:, :2000]
    return {"outer": kind, "finite": bool(torch.isfinite(y).all() and torch.isfinite(x.grad).all()),
            "causal_difference": (before - after).abs().max().item(), "first_pass_seconds": round(seconds, 2)}


def check_attention():
    torch.manual_seed(0)
    lengths = [700, 1, 300, 1100]
    starts = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    q, k, v = (torch.randn(sum(lengths), 5, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    packed = causal_within_rows(q, k, v, starts, max(lengths))
    worst = 0.0
    for begin, end in zip(starts[:-1].tolist(), starts[1:].tolist()):
        row = [t[begin:end].transpose(0, 1)[None] for t in (q, k, v)]
        alone = F.scaled_dot_product_attention(*row, is_causal=True)[0].transpose(0, 1)
        worst = max(worst, (packed[begin:end].float() - alone.float()).abs().max().item())
    return {"packed_attention_difference": worst}


def main():
    attention = check_attention()
    print(json.dumps(attention), flush=True)
    for kind in sys.argv[1:] or KINDS:
        try:
            print(json.dumps(check_outer(kind)), flush=True)
        except Exception as error:  # reported, and its model is then skipped at the timing
            print(json.dumps({"outer": kind, "error": f"{type(error).__name__}: {error}"[:500]}), flush=True)
    if attention["packed_attention_difference"] > 0.05:
        raise SystemExit("packed attention differs from PyTorch's")


if __name__ == "__main__":
    main()
