"""Checks on this GPU that a way of running predicts and learns as PyTorch's default does.

The way is one of attention, `stream` for keeping the values through the layers in bfloat16, or `documents` for
attention within documents through PyTorch's varlen attention, against the same model with a mask.
Exits with 0 if it does, and with another number if it does not or cannot run here.
Usage: attention_check.py WAY
"""

import sys

import torch

import model as model_module
from model import GPT, GPTConfig, choose_attention

LOGITS_TOLERANCE = 0.05
LEAST_GRADIENT_SIMILARITY = 0.99


def fixed_batch():
    return torch.randint(0, 512, (4, 513), generator=torch.Generator().manual_seed(1)).cuda()


def outputs(model, batch=None):
    """The logits and gradients of a fixed batch."""
    batch = fixed_batch() if batch is None else batch
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


def check_documents():
    """Attention within documents, each beginning with token 0, through varlen attention and through a mask."""
    torch.manual_seed(0)
    model = GPT(GPTConfig(block_size=512, vocab_size=512, n_layer=2, n_head=5, n_embd=640, bias=False,
                          value_embeddings=True, rotary=True, rms_norm=True, qk_norm=True, relu_squared=True,
                          separate_embeddings=True, residual_weights=True, logit_softcap=15.0, key_offset=True,
                          document_bos=0)).cuda()
    batch = fixed_batch()
    # documents of uneven lengths, one of them crossing from the first row's end into nothing
    for row, heads in enumerate(([0, 40, 300], [0, 7, 8, 200, 480], [0], [0, 256])):
        batch[row, heads] = 0
    if model_module.varlen_attn is None:
        print("documents cannot run here")
        return 2
    through_varlen = outputs(model, batch)
    kept = model_module.varlen_attn
    model_module.varlen_attn = None
    try:
        through_mask = outputs(model, batch)
    finally:
        model_module.varlen_attn = kept
    if not agree(through_varlen, through_mask):
        print("documents through varlen attention do not agree with the mask")
        return 1
    print("documents through varlen attention agree with the mask")
    return 0


def main():
    way = sys.argv[1]
    if way == "documents":
        return check_documents()
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
