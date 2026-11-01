"""Checks on this GPU that a way of running attention predicts and learns as PyTorch's default does.

Exits with 0 if it does, and with another number if it does not or cannot run here.
Usage: attention_check.py ATTENTION
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
    return logits.float().detach(), {name: p.grad.clone() for name, p in model.named_parameters()}


def agree(first, second):
    if not torch.allclose(first[0], second[0], atol=LOGITS_TOLERANCE):
        return False
    return all(
        torch.nn.functional.cosine_similarity(gradient.float().flatten(), second[1][name].float().flatten(), dim=0)
        > LEAST_GRADIENT_SIMILARITY
        for name, gradient in first[1].items()
    )


def main():
    attention = sys.argv[1]
    torch.manual_seed(0)
    model = GPT(GPTConfig(block_size=512, vocab_size=512, n_layer=2, n_head=5, n_embd=640, bias=False,
                          value_embeddings=True)).cuda()
    default = outputs(model)
    used = choose_attention(model, attention)
    if used != attention:
        print(f"{attention} cannot run here")
        return 2
    if not agree(default, outputs(model)):
        print(f"{attention} does not agree with PyTorch's default")
        return 1
    print(f"{attention} agrees with PyTorch's default")
    return 0


if __name__ == "__main__":
    sys.exit(main())
