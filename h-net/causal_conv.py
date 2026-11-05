"""The short causal convolution of Mamba-2's fused path, in Triton, for where causal-conv1d has no build.

mamba_ssm's fused Mamba-2 (mamba_split_conv1d_scan_combined) runs its convolution through causal-conv1d's CUDA
functions, which have no build for the PyTorch that research's image holds; without them Mamba-2 falls back to a
path that keeps the convolution's output and the scan's inputs for the backward pass. The two functions here take
the arguments of causal-conv1d's causal_conv1d_fwd_function and causal_conv1d_bwd_function, as the fused path calls
them, and install() puts them where it looks for them. The backward pass recomputes the convolution from its input.

Each output is out[b, c, t] = act(bias[c] + sum_k weight[c, k] x[b, c, t - (width - 1) + k]), with x zero before
the row and act SiLU or none. Width is at most 4, Mamba-2's.
"""

import os

import torch
import triton
import triton.language as tl

TAPS = 4  # the widest convolution
SPAN = 256  # the positions that one program of the backward pass sums the weights' gradients over
# Triton's interpreter cannot time configurations, so there each kernel takes its first.
INTERPRETED = os.environ.get("TRITON_INTERPRET") == "1"


def configs(options):
    chosen = options[:1] if INTERPRETED else options
    return [triton.Config({"BLOCK_T": t, "BLOCK_C": c}, num_warps=warps) for t, c, warps in chosen]


# Tiles that hold their values in registers on an H100; the backward pass's first spills none, the others a few
# bytes, against about 24 KB per thread for a tile of 64 by 128 positions and channels in 4 warps.
@triton.autotune(configs=configs([(32, 128, 4), (64, 64, 4), (128, 64, 8), (32, 64, 4), (64, 128, 4)]),
                 key=["channels", "WIDTH", "SILU"])
@triton.jit
def _forward(x_ptr, weight_ptr, bias_ptr, out_ptr, length, channels,
             stride_xb, stride_xc, stride_xt, stride_wc, stride_wk, stride_ob, stride_oc, stride_ot,
             WIDTH: tl.constexpr, HAS_BIAS: tl.constexpr, SILU: tl.constexpr,
             BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr):
    t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    row = tl.program_id(2).to(tl.int64)
    c_in = c < channels
    total = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
    if HAS_BIAS:
        total += tl.load(bias_ptr + c, mask=c_in, other=0.0).to(tl.float32)[None, :]
    x_row = x_ptr + row * stride_xb
    for k in tl.static_range(WIDTH):
        source = t - (WIDTH - 1) + k
        inside = (source >= 0) & (source < length)
        x = tl.load(x_row + source[:, None] * stride_xt + c[None, :] * stride_xc,
                    mask=inside[:, None] & c_in[None, :], other=0.0)
        w = tl.load(weight_ptr + c * stride_wc + k * stride_wk, mask=c_in, other=0.0).to(tl.float32)
        total += x.to(tl.float32) * w[None, :]
    if SILU:
        total = total * tl.sigmoid(total)
    tl.store(out_ptr + row * stride_ob + t[:, None] * stride_ot + c[None, :] * stride_oc,
             total.to(out_ptr.dtype.element_ty), mask=(t < length)[:, None] & c_in[None, :])


@triton.autotune(configs=configs([(16, 64, 4), (32, 64, 4), (32, 128, 8), (16, 128, 4), (64, 64, 8)]),
                 key=["channels", "WIDTH", "SILU"])
@triton.jit
def _backward(x_ptr, weight_ptr, bias_ptr, dout_ptr, dx_ptr, dweight_ptr, dbias_ptr, length, channels, spans,
              stride_xb, stride_xc, stride_xt, stride_wc, stride_wk, stride_gb, stride_gc, stride_gt,
              stride_db, stride_dc, stride_dt,
              WIDTH: tl.constexpr, HAS_BIAS: tl.constexpr, SILU: tl.constexpr, SPAN_T: tl.constexpr,
              TAPS_P2: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr):
    """dx over SPAN_T positions, a block at a time, and their share of dweight and dbias, written to this program's
    row of the partial sums. The gradient at the convolution's input needs the gradient before the activation at the
    next width - 1 positions too, so each block recomputes the convolution there."""
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    row = tl.program_id(2).to(tl.int64)
    c_in = c < channels
    x_row = x_ptr + row * stride_xb
    dout_row = dout_ptr + row * stride_gb
    part = row * spans + tl.program_id(0)
    bias = tl.zeros((BLOCK_C,), dtype=tl.float32)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + c, mask=c_in, other=0.0).to(tl.float32)
    taps = tl.arange(0, TAPS_P2)
    dweight = tl.zeros((TAPS_P2, BLOCK_C), dtype=tl.float32)
    dbias = tl.zeros((BLOCK_C,), dtype=tl.float32)
    for first in range(0, SPAN_T, BLOCK_T):
        t = tl.program_id(0) * SPAN_T + first + tl.arange(0, BLOCK_T)
        dx = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
        for j in tl.static_range(WIDTH):
            # The gradient before the activation at positions t + j, which tap WIDTH - 1 - j brings back to t.
            at = t + j
            grad = tl.load(dout_row + at[:, None] * stride_gt + c[None, :] * stride_gc,
                           mask=(at < length)[:, None] & c_in[None, :], other=0.0).to(tl.float32)
            if SILU:
                pre = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32) + bias[None, :]
                for k in tl.static_range(WIDTH):
                    source = at - (WIDTH - 1) + k
                    inside = (source >= 0) & (source < length)
                    x = tl.load(x_row + source[:, None] * stride_xt + c[None, :] * stride_xc,
                                mask=inside[:, None] & c_in[None, :], other=0.0)
                    w = tl.load(weight_ptr + c * stride_wc + k * stride_wk, mask=c_in, other=0.0).to(tl.float32)
                    pre += x.to(tl.float32) * w[None, :]
                sigmoid = tl.sigmoid(pre)
                grad = grad * sigmoid * (1 + pre * (1 - sigmoid))
            tap = tl.load(weight_ptr + c * stride_wc + (WIDTH - 1 - j) * stride_wk, mask=c_in, other=0.0)
            dx += grad * tap.to(tl.float32)[None, :]
            if j == 0:
                for k in tl.static_range(WIDTH):
                    source = t - (WIDTH - 1) + k
                    inside = (source >= 0) & (source < length)
                    x = tl.load(x_row + source[:, None] * stride_xt + c[None, :] * stride_xc,
                                mask=inside[:, None] & c_in[None, :], other=0.0)
                    share = tl.sum(grad * x.to(tl.float32), axis=0)
                    dweight += tl.where(taps[:, None] == k, share[None, :], 0.0)
                dbias += tl.sum(grad, axis=0)
        tl.store(dx_ptr + row * stride_db + t[:, None] * stride_dt + c[None, :] * stride_dc,
                 dx.to(dx_ptr.dtype.element_ty), mask=(t < length)[:, None] & c_in[None, :])
    tl.store(dweight_ptr + (part * WIDTH + taps[:, None]) * channels + c[None, :], dweight,
             mask=(taps[:, None] < WIDTH) & c_in[None, :])
    if HAS_BIAS:
        tl.store(dbias_ptr + part * channels + c, dbias, mask=c_in)


def _check(weight, seq_idx, initial_states, final_states):
    assert seq_idx is None and initial_states is None and final_states is None, \
        "this convolution runs whole rows from zero, without sequence numbers or states"
    assert weight.size(1) <= TAPS, f"this convolution is at most {TAPS} wide"


def convolve_forward(x, weight, bias, seq_idx, initial_states, final_states_out, silu_activation):
    """causal_conv1d_fwd_function: x of shape (rows, channels, length), weight (channels, width), bias (channels,)
    or None. The output has x's shape with the channels adjacent in memory, as the fused path reads it next."""
    _check(weight, seq_idx, initial_states, final_states_out)
    rows, channels, length = x.shape
    out = torch.empty(rows, length, channels, dtype=x.dtype, device=x.device).transpose(1, 2)
    grid = lambda meta: (triton.cdiv(length, meta["BLOCK_T"]), triton.cdiv(channels, meta["BLOCK_C"]), rows)  # noqa: E731
    _forward[grid](x, weight, bias if bias is not None else weight, out, length, channels,
                   *x.stride(), *weight.stride(), *out.stride(),
                   WIDTH=weight.size(1), HAS_BIAS=bias is not None, SILU=bool(silu_activation))
    return out


def convolve_backward(x, weight, bias, dout, seq_idx, initial_states, dfinal_states, dx, return_dinitial_states,
                      silu_activation):
    """causal_conv1d_bwd_function: returns dx (written into the given dx if there is one), dweight, dbias and no
    gradient for the initial states."""
    _check(weight, seq_idx, initial_states, dfinal_states)
    assert not return_dinitial_states
    rows, channels, length = x.shape
    width = weight.size(1)
    if dx is None:
        dx = torch.empty(rows, length, channels, dtype=x.dtype, device=x.device).transpose(1, 2)
    spans = triton.cdiv(length, SPAN)
    dweight = torch.empty(rows * spans, width, channels, dtype=torch.float32, device=x.device)
    dbias = torch.empty(rows * spans, channels, dtype=torch.float32, device=x.device) if bias is not None else None
    grid = lambda meta: (spans, triton.cdiv(channels, meta["BLOCK_C"]), rows)  # noqa: E731
    _backward[grid](x, weight, bias if bias is not None else weight, dout, dx, dweight,
                    dbias if dbias is not None else dweight, length, channels, spans,
                    *x.stride(), *weight.stride(), *dout.stride(), *dx.stride(),
                    WIDTH=width, HAS_BIAS=bias is not None, SILU=bool(silu_activation), SPAN_T=SPAN, TAPS_P2=TAPS)
    dweight = dweight.sum(dim=0).t().to(weight.dtype)
    dbias = dbias.sum(dim=0).to(bias.dtype) if bias is not None else None
    return dx, dweight, dbias, None


def install():
    """Gives mamba_ssm's fused Mamba-2 these functions where causal-conv1d's are missing; returns whether it now
    has a convolution for its fused path."""
    from mamba_ssm.ops.triton import ssd_combined

    if ssd_combined.causal_conv1d_fwd_function is None:
        ssd_combined.causal_conv1d_fwd_function = convolve_forward
        ssd_combined.causal_conv1d_bwd_function = convolve_backward
    return ssd_combined.causal_conv1d_fwd_function is not None
