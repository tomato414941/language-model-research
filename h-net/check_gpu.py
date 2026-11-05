"""Checks on the rented GPU before anything costs much: that each kind of outer layer runs forward and backward in
bfloat16 with finite values and reads only the bytes before each position, that packed attention gives what
PyTorch's attention gives row by row, and that Mamba-2's fused path, with causal_conv.py's convolution, gives what
its path with PyTorch's convolution gives. Prints a line for each, and fails if packed attention or the fused path
differs, since a model would then train on them; an outer layer that fails otherwise fails its timing too, and its
model is skipped.

Usage: python h-net/check_gpu.py [KIND...], with KIND among mamba2, mamba3 and gated_deltanet, all of them if none
"""

import copy
import json
import sys
import time

import torch
import torch.nn.functional as F

sys.path[:0] = [".", "h-net"]
from hnet_model import OuterLayer, causal_within_rows, outer_mixer  # noqa: E402

KINDS = ("mamba2", "mamba3", "gated_deltanet")
WIDTH = 256  # h-net/configs/hnet.py's outer_width
# The largest difference between the two paths, as a share of the size of what they compare: rounding to bfloat16
# differs by well under 1%, and a wrong convolution by tens of percent.
MOST_FUSED_DIFFERENCE = 0.05


def check_outer(kind):
    torch.manual_seed(0)
    layer = OuterLayer(kind, WIDTH, 0).cuda()
    x = torch.randn(4, 3328, WIDTH, device="cuda", requires_grad=True)
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


def check_fused_mamba2():
    """The largest difference between the fused and unfused paths' outputs and gradients, each relative to its size."""
    torch.manual_seed(0)
    fused = outer_mixer("mamba2", WIDTH, 0).cuda()
    unfused = copy.deepcopy(fused)
    unfused.use_mem_eff_path = False
    x = torch.randn(2, 3840, WIDTH, device="cuda")
    results = []
    for layer in (fused, unfused):
        given = x.clone().requires_grad_()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = layer(given)
        y.float().square().mean().backward()
        results.append({"output": y.float(), "input": given.grad, **{name: p.grad for name, p in layer.named_parameters()}})
    differences = {name: ((results[0][name] - value).norm() / value.norm().clamp_min(1e-12)).item()
                   for name, value in results[1].items()}
    worst = max(differences, key=differences.get)
    return {"fused_mamba2_difference": differences[worst], "at": worst}


def main():
    attention = check_attention()
    print(json.dumps(attention), flush=True)
    kinds = sys.argv[1:] or KINDS
    fused = check_fused_mamba2() if "mamba2" in kinds else None
    if fused is not None:
        print(json.dumps(fused), flush=True)
    for kind in kinds:
        try:
            print(json.dumps(check_outer(kind)), flush=True)
        except Exception as error:  # reported, and its model is then skipped at the timing
            print(json.dumps({"outer": kind, "error": f"{type(error).__name__}: {error}"[:500]}), flush=True)
    if attention["packed_attention_difference"] > 0.05:
        raise SystemExit("packed attention differs from PyTorch's")
    if fused is not None and fused["fused_mamba2_difference"] > MOST_FUSED_DIFFERENCE:
        raise SystemExit("Mamba-2's fused path differs from its unfused path")


if __name__ == "__main__":
    main()
