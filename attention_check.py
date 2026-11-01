"""Checks on this GPU that a way of running predicts and learns as PyTorch's default does.

The way is one of attention, or `stream` for keeping the values through the layers in bfloat16.
Exits with 0 if it does, and with another number if it does not or cannot run here.
Usage: attention_check.py WAY
"""

import sys

import torch

from model import GPT, GPTConfig, choose_attention

LOGITS_TOLERANCE = 0.05
LEAST_GRADIENT_SIMILARITY = 0.99


def outputs(model):
    """The logits and gradients of a fixed batch."""
    batch = torch.randint(0, 512, (4, 513), generator=torch.Generator().manual_seed(1)).cuda()
    model.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, loss = model(batch[:, :-1], batch[:, 1:].contiguous())
    loss.backward()
    # The norm takes away the size of the stream, so the gradients of the weights of one number are mostly rounding.
    return logits.float().detach(), {name: p.grad.clone() for name, p in model.named_parameters() if p.dim() > 1}


def agree(first, second):
    if not torch.allclose(first[0], second[0], atol=LOGITS_TOLERANCE):
        return False
    return all(
        torch.nn.functional.cosine_similarity(gradient.float().flatten(), second[1][name].float().flatten(), dim=0)
        > LEAST_GRADIENT_SIMILARITY
        for name, gradient in first[1].items()
    )


def main():
    way = sys.argv[1]
    torch.manual_seed(0)
    # The second bundle's shape, except for its outputs at zero, which would give most weights no first gradient.
    model = GPT(GPTConfig(block_size=512, vocab_size=512, n_layer=2, n_head=5, n_embd=640, bias=False,
                          value_embeddings=True, rotary=True, rms_norm=True, qk_norm=True, relu_squared=True,
                          separate_embeddings=True, residual_weights=True, logit_softcap=15.0)).cuda()
    default = outputs(model)
    if way == "stream":
        model.keep_stream_in(torch.bfloat16)
    elif choose_attention(model, way) != way:
        print(f"{way} cannot run here")
        return 2
    if not agree(default, outputs(model)):
        print(f"{way} does not agree with PyTorch's default")
        return 1
    print(f"{way} agrees with PyTorch's default")
    return 0


if __name__ == "__main__":
    sys.exit(main())
