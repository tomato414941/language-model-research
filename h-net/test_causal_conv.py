"""Checks of causal_conv.py against PyTorch's convolution, on a GPU, or on the CPU in Triton's interpreter, which
TRITON_INTERPRET=1 turns on only if it is set before Python starts, since importing PyTorch's attention loads Triton:

Run from the repository's root: TRITON_INTERPRET=1 python -m unittest discover -s h-net
"""

import importlib.util
import os
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path[:0] = [str(Path(__file__).resolve().parent)]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RUNNABLE = importlib.util.find_spec("triton") is not None and (DEVICE == "cuda" or os.environ.get("TRITON_INTERPRET") == "1")


def pytorch_convolution(x, weight, bias):
    out = F.conv1d(x, weight[:, None, :], bias, padding=weight.size(1) - 1, groups=x.size(1))[..., : x.size(-1)]
    return F.silu(out)


def channels_last(rows, channels, length):
    """A (rows, channels, length) view of the middle of a wider (rows, length, ...) tensor, as the fused path
    passes the convolution's part of the input projection."""
    return torch.randn(rows, length, channels + 16, device=DEVICE)[..., 8 : 8 + channels].transpose(1, 2)


@unittest.skipUnless(RUNNABLE, "needs Triton, and a GPU or TRITON_INTERPRET=1")
class CausalConvolutionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        # More channels and positions than one block holds, and not a multiple of it.
        self.x = channels_last(2, 136, 100)
        self.weight = torch.randn(136, 4, device=DEVICE)
        self.bias = torch.randn(136, device=DEVICE)

    def test_computes_pytorchs_causal_convolution_and_silu(self):
        from causal_conv import convolve_forward

        out = convolve_forward(self.x, self.weight, self.bias, None, None, None, True)
        torch.testing.assert_close(out, pytorch_convolution(self.x, self.weight, self.bias), rtol=1e-4, atol=1e-4)

    def test_gives_pytorchs_gradients_and_writes_the_input_gradient_where_asked(self):
        from causal_conv import convolve_backward

        x, weight, bias = (t.clone().requires_grad_() for t in (self.x, self.weight, self.bias))
        out = pytorch_convolution(x, weight, bias)
        dout = torch.randn_like(out)
        out.backward(dout)
        given = channels_last(2, 136, 100)
        dx, dweight, dbias, _ = convolve_backward(self.x, self.weight, self.bias, dout, None, None, None, given, False,
                                                  True)
        self.assertEqual(dx.data_ptr(), given.data_ptr())
        torch.testing.assert_close(dx, x.grad, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(dweight, weight.grad, rtol=1e-4, atol=1e-3)
        torch.testing.assert_close(dbias, bias.grad, rtol=1e-4, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
