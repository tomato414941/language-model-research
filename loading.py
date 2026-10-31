"""Load a model to read the probability of a continuation, whatever the model is made of."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch


@dataclass
class LanguageModel:
    """`logits` maps a batch of token rows to each position's logits, `encode` a string to its
    tokens, `context` is the most tokens a row may hold, and `prefix` the token that begins a
    document."""

    logits: Callable
    encode: Callable
    context: int
    prefix: int
    device: str = "cpu"
    batch_tokens: int = 4096


def encode_pair(model, context, continuation):
    """The tokens of a context and its continuation, as the LM Evaluation Harness divides them."""
    # The context's trailing spaces belong to the continuation, and the words at the boundary
    # are tokenized together, so the continuation is what follows the context's own tokens.
    spaces = len(context) - len(context.rstrip())
    if spaces:
        continuation = context[-spaces:] + continuation
        context = context[:-spaces]
    before = [model.prefix, *model.encode(context)]
    whole = [model.prefix, *model.encode(context + continuation)]
    return before, whole[len(before) :]


@torch.inference_mode()
def loglikelihoods(model, requests):
    """Each (context tokens, continuation tokens) request's log-probability and whether the model
    would have written that continuation itself."""
    prepared = []
    for index, (context, continuation) in enumerate(requests):
        if not context or not 0 < len(continuation) <= model.context:
            raise ValueError("each continuation must fit within the context length")
        tokens = (context + continuation)[-(model.context + 1) :]
        prepared.append((index, tokens, len(continuation)))
    # Rows of like length share a batch, and padding after a row's end cannot reach it.
    prepared.sort(key=lambda request: -len(request[1]))
    results = [None] * len(requests)
    start = 0
    while start < len(prepared):
        width = len(prepared[start][1]) - 1
        batch = prepared[start : start + max(1, model.batch_tokens // width)]
        start += len(batch)
        inputs = torch.full((len(batch), width), model.prefix, dtype=torch.long)
        for row, (_, tokens, _) in enumerate(batch):
            inputs[row, : len(tokens) - 1] = torch.tensor(tokens[:-1])
        # Summed in 32 bits: in bfloat16 the sums of two close choices can come out equal, and a
        # tie goes to the first choice.
        scores = model.logits(inputs.to(model.device)).float().log_softmax(-1)
        for row, (index, tokens, length) in enumerate(batch):
            end = len(tokens) - 1
            targets = torch.tensor(tokens[-length:], device=model.device)
            chosen = scores[row, end - length : end]
            results[index] = (
                chosen.gather(-1, targets[:, None]).sum().item(),
                bool((chosen.argmax(-1) == targets).all().item()),
            )
    return results


def huggingface_model(path, device, precision, batch_tokens):
    """A model in Hugging Face's format, with the tokenizer saved beside it."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    network = AutoModelForCausalLM.from_pretrained(path, dtype=getattr(torch, precision)).to(device)
    network.eval()
    return LanguageModel(
        logits=lambda tokens: network(tokens).logits,
        encode=lambda text: tokenizer.encode(text, add_special_tokens=False),
        context=network.config.max_position_embeddings,
        prefix=tokenizer.bos_token_id,
        device=device,
        batch_tokens=batch_tokens,
    )


def nanogpt_model(directory, tokenizer_path, device, precision, batch_tokens):
    """A checkpoint that train.py saved, with the tokenizer that prepare.py trained."""
    from tokenizers import Tokenizer

    from model import GPT, GPTConfig
    from prepare import BOS

    checkpoint = torch.load(Path(directory) / "ckpt.pt", map_location=device)
    network = GPT(GPTConfig(**checkpoint["model_args"]))
    network.load_state_dict({key.removeprefix("_orig_mod."): value for key, value in checkpoint["model"].items()})
    network.to(device, dtype=getattr(torch, precision))
    network.eval()
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return LanguageModel(
        # Given targets, nanoGPT's model returns every position's logits rather than the last one's.
        logits=lambda tokens: network(tokens, tokens)[0],
        encode=lambda text: tokenizer.encode(text, add_special_tokens=False).ids,
        context=network.config.block_size,
        prefix=tokenizer.token_to_id(BOS),
        device=device,
        batch_tokens=batch_tokens,
    )
