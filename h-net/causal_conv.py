"""The short causal convolution of Mamba-2's fused path, in Triton, for where causal-conv1d has no build.

mamba_ssm's fused Mamba-2 (mamba_split_conv1d_scan_combined) runs its convolution through causal-conv1d's CUDA
functions, which have no build for the PyTorch that research's image holds; without them Mamba-2 falls back to a
path that keeps the convolution's output and the scan's inputs for the backward pass. The two functions here take
the arguments of causal-conv1d's causal_conv1d_fwd_function and causal_conv1d_bwd_function, as the fused path calls
them, and install() puts them where it looks for them. The backward pass recomputes the convolution from its input.

Each output is out[b, c, t] = act(bias[c] + sum_k weight[c, k] x[b, c, t - (width - 1) + k]), with x zero before
the row and act SiLU or none.
"""

import torch
import triton
import triton.language as tl

BLOCK_LENGTH = 64
BLOCK_CHANNELS = 128


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


@triton.jit
def _backward(x_ptr, weight_ptr, bias_ptr, dout_ptr, dx_ptr, dweight_ptr, dbias_ptr, length, channels, blocks,
              stride_xb, stride_xc, stride_xt, stride_wc, stride_wk, stride_gb, stride_gc, stride_gt,
              stride_db, stride_dc, stride_dt,
              WIDTH: tl.constexpr, HAS_BIAS: tl.constexpr, SILU: tl.constexpr,
              BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr):
    """dx at this block's positions, and this block's share of dweight and dbias, written to its own row of the
    partial sums. The gradient at the convolution's input needs the gradient before the activation at the next
    width - 1 positions too, so each block recomputes the convolution there."""
    t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    row = tl.program_id(2).to(tl.int64)
    c_in = c < channels
    x_row = x_ptr + row * stride_xb
    dout_row = dout_ptr + row * stride_gb
    part = row * blocks + tl.program_id(0)
    bias = tl.zeros((BLOCK_C,), dtype=tl.float32)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + c, mask=c_in, other=0.0).to(tl.float32)
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
            grad = tl.where((t < length)[:, None], grad, 0.0)
            for k in tl.static_range(WIDTH):
                source = t - (WIDTH - 1) + k
                inside = (source >= 0) & (source < length)
                x = tl.load(x_row + source[:, None] * stride_xt + c[None, :] * stride_xc,
                            mask=inside[:, None] & c_in[None, :], other=0.0)
                tl.store(dweight_ptr + (part * WIDTH + k) * channels + c, tl.sum(grad * x.to(tl.float32), axis=0),
                         mask=c_in)
            if HAS_BIAS:
                tl.store(dbias_ptr + part * channels + c, tl.sum(grad, axis=0), mask=c_in)
    tl.store(dx_ptr + row * stride_db + t[:, None] * stride_dt + c[None, :] * stride_dc,
             dx.to(dx_ptr.dtype.element_ty), mask=(t < length)[:, None] & c_in[None, :])


def _grid(rows, channels, length):
    return (triton.cdiv(length, BLOCK_LENGTH), triton.cdiv(channels, BLOCK_CHANNELS), rows)


def _check(seq_idx, initial_states, final_states):
    assert seq_idx is None and initial_states is None and final_states is None, \
        "this convolution runs whole rows from zero, without sequence numbers or states"


def convolve_forward(x, weight, bias, seq_idx, initial_states, final_states_out, silu_activation):
    """causal_conv1d_fwd_function: x of shape (rows, channels, length), weight (channels, width), bias (channels,)
    or None. The output has x's shape with the channels adjacent in memory, as the fused path reads it next."""
    _check(seq_idx, initial_states, final_states_out)
    rows, channels, length = x.shape
    out = torch.empty(rows, length, channels, dtype=x.dtype, device=x.device).transpose(1, 2)
    _forward[_grid(rows, channels, length)](
        x, weight, bias if bias is not None else weight, out, length, channels,
        *x.stride(), *weight.stride(), *out.stride(),
        WIDTH=weight.size(1), HAS_BIAS=bias is not None, SILU=bool(silu_activation),
        BLOCK_T=BLOCK_LENGTH, BLOCK_C=BLOCK_CHANNELS)
    return out


def convolve_backward(x, weight, bias, dout, seq_idx, initial_states, dfinal_states, dx, return_dinitial_states,
                      silu_activation):
    """causal_conv1d_bwd_function: returns dx (written into the given dx if there is one), dweight, dbias and no
    gradient for the initial states."""
    _check(seq_idx, initial_states, dfinal_states)
    assert not return_dinitial_states
    rows, channels, length = x.shape
    width = weight.size(1)
    if dx is None:
        dx = torch.empty(rows, length, channels, dtype=x.dtype, device=x.device).transpose(1, 2)
    blocks = triton.cdiv(length, BLOCK_LENGTH)
    dweight = torch.empty(rows * blocks, width, channels, dtype=torch.float32, device=x.device)
    dbias = torch.empty(rows * blocks, channels, dtype=torch.float32, device=x.device) if bias is not None else None
    _backward[_grid(rows, channels, length)](
        x, weight, bias if bias is not None else weight, dout, dx, dweight, dbias if dbias is not None else dweight,
        length, channels, blocks, *x.stride(), *weight.stride(), *dout.stride(), *dx.stride(),
        WIDTH=width, HAS_BIAS=bias is not None, SILU=bool(silu_activation),
        BLOCK_T=BLOCK_LENGTH, BLOCK_C=BLOCK_CHANNELS)
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
