"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    # PyTorch's attention over sequences of different lengths, which keeps each document to itself on a GPU
    from torch.nn.attention.varlen import varlen_attn
except ImportError:
    varlen_attn = None

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class RMSNorm(nn.Module):
    """ Divides by the root mean square, with no mean to subtract and no weights to learn.
    It gives back the type it was given, which autocast on a GPU would turn into float32. """

    def forward(self, input):
        return F.rms_norm(input, (input.size(-1),)).to(input.dtype)

def make_norm(config):
    return RMSNorm() if config.rms_norm else LayerNorm(config.n_embd, bias=config.bias)

def rotary_tables(length, head_dim, base=10000, still_pairs=0):
    """The cosine and sine that turn each pair of a head's channels at each position, as in karpathy/nanochat.
    The last `still_pairs` pairs do not turn, and the others spread over the whole range of rates."""
    turning = head_dim // 2 - still_pairs
    rates = torch.cat([1.0 / (base ** (torch.arange(turning).float() / turning)), torch.zeros(still_pairs)])
    angles = torch.outer(torch.arange(length).float(), rates)
    return angles.cos()[None, :, None, :], angles.sin()[None, :, None, :]

def still_channels(head_dim):
    """The channels of a head that the key offset moves: the second half of each half, which rotate() turns by the
    last half of the rates, the ones that are zero with the key offset."""
    channel = torch.arange(head_dim) % (head_dim // 2)
    return channel >= head_dim // 4

def offset_keys(k, still, first=None):
    """Gives each position the still channels of the key before it, so that one layer can find what followed an
    earlier copy of a token, as in modded-nanogpt's record 49. The first position keeps its own, and so does each
    position where `first` (B, T) is true, the head of a document when attention stays within one. k is (B, T, nh, hs)."""
    moved = torch.cat([k[:, :1], k[:, :-1]], dim=1)
    if first is not None:
        moved = torch.where(first[:, :, None, None], k, moved)
    return torch.where(still, moved, k)

BIGRAM_ROWS_PER_TOKEN = 5 # as in karpathy/nanochat and modded-nanogpt's record 62

def bigram_rows(idx, rows, documents=None):
    """Each position's row in a table of `rows`: a hash of its token and the one before it, as in modded-nanogpt's
    record 62. The first position of a row has no token before it and takes the last row, and so does the head of
    a document when `documents` (document_ids) is given."""
    idx = idx.long()
    hashed = torch.bitwise_xor(36313 * idx[:, 1:], 27191 * idx[:, :-1]) % (rows - 1)
    if documents is not None:
        hashed = torch.where(documents[:, 1:] == documents[:, :-1], hashed, rows - 1)
    return torch.cat([torch.full_like(idx[:, :1], rows - 1), hashed], dim=1)

def trigram_rows(idx, rows, documents=None):
    """Each position's row in a table of `rows`: a hash of its token and the two before it, as Over-Encoding
    (arXiv 2501.16975) adds 3-grams to 2-grams. A position with fewer than two tokens before it in its row, or in
    its document when `documents` is given, takes the last row."""
    idx = idx.long()
    hashed = torch.bitwise_xor(torch.bitwise_xor(36313 * idx[:, 2:], 27191 * idx[:, 1:-1]), 51497 * idx[:, :-2])
    hashed = hashed % (rows - 1)
    if documents is not None:
        hashed = torch.where(documents[:, 2:] == documents[:, :-2], hashed, rows - 1)
    return torch.cat([torch.full_like(idx[:, :2], rows - 1), hashed], dim=1)

# Room for the documents of a pass in attention over variable lengths: this many per row on average. Should a
# pass hold more, the latest documents of a row share attention, as all of a row's documents did before.
SEGMENTS_PER_ROW = 8

def document_ids(idx, bos):
    """Each position's document within its row: how many BOS tokens the row has read up to and including it."""
    return torch.cumsum((idx == bos).int(), dim=1)

def segment_bounds(documents, per_row=None):
    """The flat positions at which a row or a document begins, then the end, as the cumulative sequence positions
    of attention over variable lengths. The list has a fixed length, `per_row` (SEGMENTS_PER_ROW unless given) per
    row and one, so that a compiled model takes every batch alike: the room left over repeats the end. Every row's
    beginning is kept, so that no segment runs past a row."""
    per_row = SEGMENTS_PER_ROW if per_row is None else per_row
    b, t = documents.shape
    end = b * t
    positions = torch.arange(end, device=documents.device)
    heads = torch.ones_like(documents, dtype=torch.bool)
    heads[:, 1:] = documents[:, 1:] != documents[:, :-1]
    rows = positions % t == 0
    # rows first, then documents in order, then everything else, with room for a pass shorter than the list
    keys = torch.where(rows, positions - end, torch.where(heads.flatten(), positions, 2 * end))
    room = b * per_row
    keys = torch.cat([keys, keys.new_full((max(room - end, 0),), 2 * end)])
    kept = torch.sort(keys).values[:room]
    kept = torch.where(kept < 0, kept + end, torch.clamp(kept, max=end))
    return torch.cat([torch.sort(kept).values, kept.new_full((1,), end)]).int()

def window_segments(documents, window):
    """The segments that attention keeps to while it reads within windows: each document cut where the row's
    positions cross a multiple of `window`, SkyLadder's block-wise mask over a packed row (arXiv 2503.15450). Each
    position then attends at most `window` positions back, and to all of its document once `window` covers the row.
    `window` is a tensor, so that a compiled model takes every size alike."""
    t = documents.size(1)
    return documents * (t + 1) + torch.arange(t, device=documents.device) // window

DENSE_INNER = 64 # the width of the small MLP that weighs the earlier outputs, as in modded-nanogpt's record 81
DENSE_VALUE_SCALE = 1.15 # the record's scale on the mixture added to the values

def dense_sources(n_layer):
    """The earlier outputs that each of the last two blocks mixes after it runs, by layer; -1 is the blocks' input.
    modded-nanogpt's record 81 takes these for its 11 layers: the input and the outputs of layers 7 and 9 after
    layer 9, and the input and layer 9 after layer 10."""
    last = n_layer - 1
    return {last - 1: tuple(dict.fromkeys(s for s in (-1, last - 3, last - 1) if s >= -1)), last: (-1, last - 1)}

def backout_layer(n_layer):
    """The layer after which the stream is kept, to take a share of it out before the prediction: after 8 of
    11 layers in modded-nanogpt's records 40 to 81, and after 7 of 10."""
    return round(8 * n_layer / 11) - 1

def rotate(x, cos, sin):
    """Turns x of shape (B, T, nh, hs) by its position, so that a query and a key meet by their distance alone."""
    cos, sin = cos[:, :x.size(1)].to(x.dtype), sin[:, :x.size(1)].to(x.dtype)
    half = x.size(-1) // 2
    first, second = x[..., :half], x[..., half:]
    return torch.cat([first * cos + second * sin, second * cos - first * sin], dim=-1)

VALUE_GATE_CHANNELS = 12
ATTENTION_GATE_CHANNELS = 12 # as in modded-nanogpt's record 28
DISPERSION_STRIDE = 8 # Dispersion Loss reads every eighth position, as its paper allows sampling the tokens
DISPERSION_TEMPERATURE = 1.0

def has_value_embedding(layer, n_layer):
    """Value embeddings sit on every other layer and always on the last, as in karpathy/nanochat."""
    return layer % 2 == (n_layer - 1) % 2

def has_stem_table(layer, n_layer):
    """STEM's tables sit on the layers without value embeddings: half of an even number of layers, at even intervals,
    which STEM (arXiv 2601.10639) found better than a third or all of them."""
    return not has_value_embedding(layer, n_layer)

def load_flash_attention():
    """FlashAttention-3's attention function, or None where it cannot run. Its kernels stop at Hopper GPUs."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] > 9:
        return None
    try:
        from kernels import get_kernel
        return get_kernel("kernels-community/flash-attn3", version=2).flash_attn_func
    except Exception as error:
        print(f"FlashAttention-3 could not be loaded: {error}")
        return None

def choose_attention(model, attention):
    """Makes attention run as asked on this GPU and returns what runs, 'sdpa' if the asked one cannot.

    'sdpa' leaves the choice to PyTorch, which takes cuDNN on recent GPUs. 'fa2' is the FlashAttention-2
    built into PyTorch and 'fa4' is FlashAttention-4 behind PyTorch's call, both with cuDNN turned off.
    'fa3' is called directly, since PyTorch's own route to it only finds a build made from source.
    """
    if attention == 'fa3':
        function = load_flash_attention()
        if function is None:
            return 'sdpa'
        model.use_flash_attention(function)
    elif attention == 'fa4':
        try:
            torch.nn.attention.activate_flash_attention_impl('FA4')
        except Exception as error:
            print(f"FlashAttention-4 could not be loaded: {error}")
            return 'sdpa'
    if attention in ('fa2', 'fa4'):
        torch.backends.cuda.enable_cudnn_sdp(False)
    return attention

class CausalSelfAttention(nn.Module):

    def __init__(self, config, layer):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # a gate per head on the value embedding, read from the first channels of the input
        self.value_gate = None
        if config.value_embeddings and has_value_embedding(layer, config.n_layer):
            self.value_gate = nn.Linear(VALUE_GATE_CHANNELS, config.n_head, bias=False)
        # a gate per head on its output, read from the first channels of the input, so that a head can do nothing
        self.attention_gate = None
        if config.attention_gate:
            self.attention_gate = nn.Linear(ATTENTION_GATE_CHANNELS, config.n_head, bias=False)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # positions by turning the queries and keys, in place of the table of position embeddings
        self.rotary = config.rotary
        head_dim = config.n_embd // config.n_head
        if self.rotary:
            assert head_dim % 2 == 0
            # the key offset needs channels that position does not turn, so that a key moved one place still meets
            still_pairs = head_dim // 4 if config.key_offset else 0
            cos, sin = rotary_tables(config.block_size, head_dim, still_pairs=still_pairs)
            self.register_buffer("rotary_cos", cos, persistent=False)
            self.register_buffer("rotary_sin", sin, persistent=False)
        self.key_offset = config.key_offset
        if self.key_offset:
            assert config.rotary and head_dim % 4 == 0, "the key offset moves the channels that rotation leaves still"
            self.register_buffer("still", still_channels(head_dim), persistent=False)
        # queries and keys of one size, so that only their directions decide where to attend
        self.qk_norm = RMSNorm() if config.qk_norm else None
        # set through GPT.use_flash_attention, in place of PyTorch's attention
        self.flash_attention = None
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, value_embedding=None, value_mixture=None, documents=None, bounds=None, segments=None):
        """With `documents` (document_ids), each position attends only within its document, or within its piece of
        it when `segments` (window_segments) are given. On a GPU this takes `bounds` (segment_bounds) for attention
        over variable lengths; elsewhere a mask does the same."""
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        if value_embedding is not None:
            gate = 3 * torch.sigmoid(self.value_gate(x[..., :VALUE_GATE_CHANNELS])) # (B, T, nh), range (0, 3)
            v = v + gate.repeat_interleave(C // self.n_head, dim=-1) * value_embedding
        if value_mixture is not None:
            v = v + value_mixture
        k = k.view(B, T, self.n_head, C // self.n_head) # (B, T, nh, hs)
        q = q.view(B, T, self.n_head, C // self.n_head) # (B, T, nh, hs)
        v = v.view(B, T, self.n_head, C // self.n_head) # (B, T, nh, hs)
        if self.rotary:
            q, k = rotate(q, self.rotary_cos, self.rotary_sin), rotate(k, self.rotary_cos, self.rotary_sin)
        if self.qk_norm is not None:
            q, k = self.qk_norm(q), self.qk_norm(k)
        # the pieces that attention keeps to: documents, or their pieces within windows
        kept = documents if segments is None else segments
        if self.key_offset:
            first = None
            if kept is not None:
                first = torch.ones_like(kept, dtype=torch.bool)
                first[:, 1:] = kept[:, 1:] != kept[:, :-1]
            k = offset_keys(k, self.still, first)
        if documents is not None and bounds is not None:
            # (-1, 0) is a causal window with no limit to the left
            y = varlen_attn(q.bfloat16().flatten(0, 1), k.bfloat16().flatten(0, 1), v.bfloat16().flatten(0, 1),
                            bounds, bounds, T, T, window_size=(-1, 0)).view(B, T, self.n_head, C // self.n_head)
        elif documents is not None:
            causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            mask = (kept[:, None, :, None] == kept[:, None, None, :]) & causal
            k, q, v = k.transpose(1, 2), q.transpose(1, 2), v.transpose(1, 2) # (B, nh, T, hs)
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v.to(q.dtype), attn_mask=mask).transpose(1, 2)
        elif self.flash_attention is not None:
            # All three must be bf16, as autocast makes them for PyTorch's attention.
            y = self.flash_attention(q.bfloat16(), k.bfloat16(), v.bfloat16(), causal=True) # (B, T, nh, hs)
        else:
            k, q, v = k.transpose(1, 2), q.transpose(1, 2), v.transpose(1, 2) # (B, nh, T, hs)
            # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            if self.flash:
                # efficient attention using Flash Attention CUDA kernels
                y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
            else:
                # manual implementation of attention
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            y = y.transpose(1, 2) # (B, T, nh, hs)
        if self.attention_gate is not None:
            gate = torch.sigmoid(self.attention_gate(x[..., :ATTENTION_GATE_CHANNELS])) # (B, T, nh), range (0, 1)
            y = y * gate.unsqueeze(-1).to(y.dtype)
        y = y.contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

@torch.compiler.disable
def count_candidates(table, idx):
    """The COUNT table's candidates and weights for each position, looked up outside the compiled graph."""
    return table.candidates(idx)

def dispersion_loss(vectors, temperature=DISPERSION_TEMPERATURE):
    """Dispersion Loss (arXiv 2602.00217) over the last two axes: low when a row's vectors point in different
    directions. It is the log of the mean of exp(-angle / (pi * temperature)) over the pairs of different vectors,
    the paper's log of the sum less a constant, so that it lies between -1/temperature and 0."""
    unit = F.normalize(vectors.float(), dim=-1)
    cosines = (unit @ unit.transpose(-1, -2)).clamp(-1 + 1e-4, 1 - 1e-4)
    angles = torch.arccos(cosines) / math.pi
    count = cosines.size(-1)
    others = ~torch.eye(count, dtype=torch.bool, device=vectors.device)
    return (torch.logsumexp(-angles[..., others] / temperature, dim=-1) - math.log(count * (count - 1))).mean()

class ReLUSquared(nn.Module):
    """ Squares the positive values and zeroes the others, with no weights to learn """

    def forward(self, input):
        return F.relu(input).square()

def inverse_softplus(value):
    return math.log(math.expm1(value))

class XIELU(nn.Module):
    """xIELU (arXiv 2411.13010), as Apertus runs it: alpha_p x^2 + beta x above zero, and
    alpha_n (e^x - 1) - alpha_n x + beta x below, where the layer learns alpha_p and alpha_n through softplus."""

    def __init__(self, alpha_p=0.8, alpha_n=0.8, beta=0.5, eps=-1e-6):
        super().__init__()
        self.alpha_p = nn.Parameter(torch.tensor([inverse_softplus(alpha_p)]))
        self.alpha_n = nn.Parameter(torch.tensor([inverse_softplus(alpha_n - beta)]))
        self.beta = beta
        self.eps = eps

    def forward(self, input):
        alpha_p = F.softplus(self.alpha_p).to(input.dtype)
        alpha_n = (self.beta + F.softplus(self.alpha_n)).to(input.dtype)
        negative = (torch.expm1(torch.clamp(input, max=self.eps)) - input) * alpha_n
        return torch.where(input > 0, alpha_p * input * input, negative) + self.beta * input

def make_activation(config):
    if config.xielu:
        return XIELU()
    return ReLUSquared() if config.relu_squared else nn.GELU()

class MLP(nn.Module):

    def __init__(self, config, width=None):
        super().__init__()
        width = width or 4 * config.n_embd
        self.c_fc    = nn.Linear(config.n_embd, width, bias=config.bias)
        self.activation = make_activation(config)
        self.c_proj  = nn.Linear(width, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, stem_row=None):
        """With `stem_row`, the token's row of a STEM table, which scales each of the widened values by one plus its
        own number, in the place of the up projection that STEM replaces in a gated MLP."""
        x = self.c_fc(x)
        x = self.activation(x)
        if stem_row is not None:
            x = x * (1 + stem_row.to(x.dtype))
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

# In training, the routed tokens that each expert takes, as a share of its even part; the rest are dropped, as Switch
# Transformer and ST-MoE train (arXiv 2101.03961, 2202.08906). Evaluation drops none.
EXPERT_CAPACITY = 1.25

def has_experts(layer, config):
    """Experts take the MLPs of the layers without STEM's tables, every other layer as GShard and Switch place them,
    so that each layer keeps one kind of table looked up by the token."""
    return config.experts > 0 and not (config.stem and has_stem_table(layer, config.n_layer))

class Experts(nn.Module):
    """A mixture of experts in the place of the MLP: DeepSeekMoE's fine-grained routed experts beside shared ones
    (arXiv 2401.06066). Each token goes to the experts_per_token routed experts of highest sigmoid affinity, as
    DeepSeek-V3 routes (arXiv 2412.19437), weighed by their affinities brought to a sum of one. The router runs in
    float32, as Switch Transformer's does. A bias per expert, which only the choice sees, balances the load without an
    auxiliary loss (Loss-Free Balancing, arXiv 2408.15664): `rebalance` moves it after each update by the load that
    the update's tokens put on each expert."""

    def __init__(self, config):
        super().__init__()
        width, n_embd = config.expert_width, config.n_embd
        self.count, self.chosen = config.experts, config.experts_per_token
        assert 0 < self.chosen <= self.count and width > 0
        self.router = nn.Linear(n_embd, self.count, bias=False)
        # each expert's two matrices, held as nn.Linear holds its weight (out, in)
        self.w_in = nn.Parameter(torch.empty(self.count, width, n_embd).normal_(0.0, 0.02))
        self.w_out = nn.Parameter(torch.empty(self.count, n_embd, width))
        if config.zero_output:
            torch.nn.init.zeros_(self.w_out)
        else:
            torch.nn.init.normal_(self.w_out, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        self.activation = make_activation(config)
        self.shared = MLP(config, width * config.shared_experts) if config.shared_experts else None
        self.register_buffer("balance", torch.zeros(self.count))
        self.register_buffer("load", torch.zeros(self.count), persistent=False)

    def route(self, flat):
        """Each token's chosen experts and their weights."""
        with torch.autocast(device_type=flat.device.type, enabled=False):
            affinity = torch.sigmoid(F.linear(flat.float(), self.router.weight.float()))
        chosen = torch.topk(affinity + self.balance, self.chosen, dim=-1).indices
        weights = affinity.gather(1, chosen)
        return chosen, weights / weights.sum(-1, keepdim=True)

    def forward(self, x, stem_row=None):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        chosen, weights = self.route(flat)
        if self.training:
            self.load.index_add_(0, chosen.flatten(), torch.ones(chosen.numel(), device=flat.device))
            routed = self.dropping(flat, chosen, weights)
        else:
            routed = self.dropless(flat, chosen, weights)
        if self.shared is not None:
            routed = routed + self.shared(flat)
        return routed.view(shape)

    def dropping(self, flat, chosen, weights):
        """The routed experts' outputs in fixed shapes, which a compiled model needs: each expert takes up to
        EXPERT_CAPACITY times its even share of the routed tokens, the earlier tokens first."""
        n, n_embd = flat.shape
        routes = n * self.chosen
        capacity = math.ceil(routes / self.count * EXPERT_CAPACITY)
        expert = chosen.flatten()
        order = torch.argsort(expert, stable=True)
        counts = torch.zeros(self.count, dtype=torch.long, device=flat.device).index_add_(0, expert, torch.ones_like(expert))
        starts = counts.cumsum(0) - counts
        place = torch.empty_like(expert).scatter_(0, order, torch.arange(routes, device=flat.device) - starts[expert[order]])
        kept = place < capacity
        # the last row of the experts' inputs takes the routes that do not fit, and gives back nothing
        slot = torch.where(kept, expert * capacity + place, self.count * capacity)
        token = torch.arange(n, device=flat.device).repeat_interleave(self.chosen)
        inputs = flat.new_zeros(self.count * capacity + 1, n_embd).index_copy(0, slot, flat[token])
        hidden = self.activation(torch.bmm(inputs[:-1].view(self.count, capacity, n_embd), self.w_in.mT))
        outputs = torch.bmm(hidden, self.w_out.mT).reshape(-1, n_embd)
        outputs = torch.cat([outputs, outputs.new_zeros(1, n_embd)])
        share = (weights.flatten() * kept).to(outputs.dtype)
        return (outputs[slot] * share[:, None]).view(n, self.chosen, n_embd).sum(1).to(flat.dtype)

    def dropless(self, flat, chosen, weights):
        """The routed experts' outputs for every route, an expert at a time."""
        routed = torch.zeros_like(flat)
        for expert in range(self.count):
            tokens, places = (chosen == expert).nonzero(as_tuple=True)
            if len(tokens):
                hidden = self.activation(flat[tokens] @ self.w_in[expert].mT)
                output = (hidden @ self.w_out[expert].mT) * weights[tokens, places, None]
                routed.index_add_(0, tokens, output.to(flat.dtype))
        return routed

    @torch.no_grad()
    def rebalance(self, rate):
        """Raises the bias of each expert that took fewer than the mean of the routes since the last call by `rate`,
        and lowers the others', as Loss-Free Balancing does after each batch."""
        self.balance += rate * torch.sign(self.load.mean() - self.load)
        self.load.zero_()

class Block(nn.Module):

    def __init__(self, config, layer):
        super().__init__()
        self.ln_1 = make_norm(config)
        self.attn = CausalSelfAttention(config, layer)
        self.ln_2 = make_norm(config)
        self.mlp = Experts(config) if has_experts(layer, config) else MLP(config)

    def forward(self, x, value_embedding=None, value_mixture=None, documents=None, bounds=None, stem_row=None,
                segments=None):
        x = x + self.attn(self.ln_1(x), value_embedding, value_mixture, documents, bounds, segments)
        x = x + self.mlp(self.ln_2(x), stem_row)
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    value_embeddings: bool = False # True: a table per token that adds to the values of some layers
    rotary: bool = False # True: positions by rotary embeddings, not by a table of position embeddings
    rms_norm: bool = False # True: RMSNorm with no weights to learn, not LayerNorm
    qk_norm: bool = False # True: queries and keys are normalized before they meet
    relu_squared: bool = False # True: ReLU squared in the MLP, not GELU
    zero_output: bool = False # True: the last weights of attention and the MLP start at zero
    separate_embeddings: bool = False # True: one table reads the tokens and another predicts them
    residual_weights: bool = False # True: each block takes the stream and the embedded tokens by weights it learns
    logit_softcap: float = 0.0 # above zero: the logits are squashed smoothly into plus or minus this
    count: bool = False # True: the input also takes the output table's rows of the tokens a COUNT table expects next
    dispersion: float = 0.0 # above zero: the weight of a loss that spreads the directions of each row's token vectors
    bigram_table: bool = False # True: a table looked up by each token and the one before it adds to every block's input
    attention_gate: bool = False # True: each head's output is scaled by a gate read from the first channels of the input
    key_offset: bool = False # True: half of each head's channel pairs do not turn, and the keys take them from the token before
    dense_connections: bool = False # True: the last two blocks add mixtures of earlier outputs, weighted per position (MUDD)
    backout: bool = False # True: a share of the stream after two thirds of the layers is taken out before the prediction
    bigram_rows_per_token: int = BIGRAM_ROWS_PER_TOKEN # the rows of the table of pairs, and of triples, per token
    trigram_table: bool = False # True: a table looked up by each token and the two before it adds to every block's input
    stem: bool = False # True: half of the MLPs scale their widened values by a table looked up by each token (STEM)
    xielu: bool = False # True: xIELU in the MLP, in the place of ReLU squared or GELU
    document_bos: int = -1 # at least zero: attention stays within each document, which begins with this token
    smallest_window: int = 0 # above zero: attention may keep to windows (GPT.attention_window) at least this long
    experts: int = 0 # above zero: the routed experts of the MLPs that has_experts gives a mixture of experts
    experts_per_token: int = 0 # the routed experts that each token goes to
    expert_width: int = 0 # the width of each expert's widened values
    shared_experts: int = 0 # the experts that every token goes to, as one MLP that many times as wide as an expert

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            **({} if config.rotary else dict(wpe = nn.Embedding(config.block_size, config.n_embd))),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config, layer) for layer in range(config.n_layer)]),
            ln_f = make_norm(config),
        ))
        self.value_embeds = nn.ModuleDict({
            str(layer): nn.Embedding(config.vocab_size, config.n_embd) for layer in range(config.n_layer)
            if config.value_embeddings and has_value_embedding(layer, config.n_layer)
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # how much of the stream and of the embedded tokens each block takes, as in karpathy/autoresearch
        if config.residual_weights:
            self.resid_weights = nn.Parameter(torch.ones(config.n_layer))
            self.x0_weights = nn.Parameter(torch.full((config.n_layer,), 0.1))
        # how much the COUNT table's view of what comes next adds to the input, as a weight it learns
        if config.count:
            self.count_gate = nn.Parameter(torch.full((1,), 0.1))
        # a table of hashed pairs of tokens, which adds to each block's input by a weight per layer
        if config.bigram_table:
            self.bigram_table = nn.Embedding(config.bigram_rows_per_token * config.vocab_size, config.n_embd)
            self.bigram_weights = nn.Parameter(torch.full((config.n_layer,), 0.1))
        # a table of hashed triples of tokens, which adds to each block's input like the table of pairs
        if config.trigram_table:
            self.trigram_table = nn.Embedding(config.bigram_rows_per_token * config.vocab_size, config.n_embd)
            self.trigram_weights = nn.Parameter(torch.full((config.n_layer,), 0.1))
        # STEM's tables, one row per token as wide as the widened values of the MLP
        self.stem_tables = nn.ModuleDict({
            str(layer): nn.Embedding(config.vocab_size, 4 * config.n_embd) for layer in range(config.n_layer)
            if config.stem and has_stem_table(layer, config.n_layer)
        })
        # the small MLPs that weigh, at each position, the earlier outputs that the last two blocks mix
        if config.dense_connections:
            assert config.n_layer >= 3, "the last two blocks mix outputs from before them"
            self.dense_sources = dense_sources(config.n_layer)
            most = max(len(sources) for sources in self.dense_sources.values())
            self.dense_in = nn.Parameter(torch.empty(2, DENSE_INNER, config.n_embd))
            for weight in self.dense_in:
                torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            # two ways to add each mixture, to the values of the next block and to the stream, starting at nothing
            self.dense_out = nn.Parameter(torch.zeros(2, DENSE_INNER, 2, most))
            self.dense_bias = nn.Parameter(torch.zeros(2, 2, most))
        # how much of the stream after two thirds of the layers is taken out before the prediction
        if config.backout:
            self.backout_weight = nn.Parameter(torch.full((1,), 0.5))
        # set through GPT.use_count_table
        self.count_table = None
        # how far back attention reads, which training may shorten (SkyLadder); the whole context unless set
        if config.smallest_window:
            assert config.document_bos >= 0, "windows cut documents, so attention must keep to documents"
        self.register_buffer("attention_window", torch.tensor(config.block_size), persistent=False)
        # only the direction of each token's vector in a separate reading table goes on into the blocks
        self.input_norm = RMSNorm() if config.separate_embeddings else None
        # set through GPT.keep_stream_in
        self.stream_dtype = None
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        if not config.separate_embeddings:
            self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # a separate reading table starts large, in step with its large learning rate, and predictions start near zero
        if config.separate_embeddings:
            torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
            torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        # apply special scaled init to the residual projections, per GPT-2 paper, or start them at zero
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                if config.zero_output:
                    torch.nn.init.zeros_(p)
                else:
                    torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        # value gates start small and positive, so that each table has a little effect from the start
        for block in self.transformer.h:
            if block.attn.value_gate is not None:
                torch.nn.init.uniform_(block.attn.value_gate.weight, 0.0, 0.02)
            # attention gates start at zero, half open everywhere, as in modded-nanogpt's record 28
            if block.attn.attention_gate is not None:
                torch.nn.init.zeros_(block.attn.attention_gate.weight)
        # the tables of pairs and triples start at zero, so that they add nothing until they learn
        if config.bigram_table:
            torch.nn.init.zeros_(self.bigram_table.weight)
        if config.trigram_table:
            torch.nn.init.zeros_(self.trigram_table.weight)
        # STEM's tables start at zero, so that each MLP starts as it would without one
        for table in self.stem_tables.values():
            torch.nn.init.zeros_(table.weight)

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def use_flash_attention(self, function):
        """Sends attention through this function. It is no part of the weights, so it is not saved with them."""
        for block in self.transformer.h:
            block.attn.flash_attention = function

    def use_count_table(self, table):
        """Looks up this COUNT table (count_table.py) at every position. It is no part of the weights, so it is not
        saved with them."""
        self.count_table = table

    def count_input(self, idx):
        """The output table's rows of the tokens that the COUNT table expects after each position's histories,
        weighted by their scores and averaged over the history lengths, brought to one size."""
        assert self.count_table is not None, "a COUNT model looks up its table; see GPT.use_count_table"
        tokens, weights = count_candidates(self.count_table, idx)
        b, t, k = tokens.shape
        weight = self.lm_head.weight
        rows = F.embedding_bag(tokens.view(-1, k), weight, per_sample_weights=weights.view(-1, k).to(weight.dtype),
                               mode='sum')
        return F.rms_norm(rows, (rows.size(-1),)).view(b, t, -1)

    def keep_stream_in(self, dtype):
        """Holds the values that pass through the layers in this type. It is no part of the weights, so it is
        not saved with them. Linear layers take the type only under autocast."""
        self.stream_dtype = dtype

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position and value embeddings get subtracted.
        The token embeddings do too when they are separate. Otherwise, due to the parameter
        sharing these params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            if not self.config.rotary:
                n_params -= self.transformer.wpe.weight.numel()
            if self.config.separate_embeddings:
                n_params -= self.transformer.wte.weight.numel()
            n_params -= sum(p.numel() for p in self.value_embeds.parameters())
            n_params -= sum(p.numel() for p in self.stem_tables.parameters())
            if self.config.bigram_table:
                n_params -= self.bigram_table.weight.numel()
            if self.config.trigram_table:
                n_params -= self.trigram_table.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def mix_earlier_outputs(self, layer, x, outputs):
        """MUDD (arXiv 2502.12170) as modded-nanogpt's record 81 cuts it down: after one of the last two blocks, a
        small MLP of the stream weighs some earlier outputs at each position. One mixture adds to the stream, and
        after the second last block another goes to the values of the last. Both start at zero."""
        sources = self.dense_sources[layer]
        count = len(sources)
        which = layer - (self.config.n_layer - 2)
        hidden = F.gelu(F.linear(x, self.dense_in[which].to(x.dtype)))
        weights = torch.einsum('bti,icl->btcl', hidden, self.dense_out[which, :, :, :count].to(x.dtype))
        weights = (weights + self.dense_bias[which, :, :count].to(x.dtype)) * (0.2 / math.sqrt(count))
        stacked = torch.stack([outputs[source] for source in sources], dim=-1) # (b, t, n_embd, count)
        x = x + (stacked * weights[:, :, 1:2]).sum(-1)
        if layer == self.config.n_layer - 1:
            return x, None
        return x, DENSE_VALUE_SCALE * (stacked * weights[:, :, 0:1]).sum(-1)

    def cap_logits(self, logits):
        """Squashes the logits as the config asks, in float32, since tanh loses too much in bfloat16."""
        cap = self.config.logit_softcap
        return cap * torch.tanh(logits.float() / cap) if cap > 0 else logits

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        if not self.config.rotary:
            pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
            x = x + self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        if self.input_norm is not None:
            x = self.input_norm(x)
        if self.config.count:
            x = x + self.count_gate * self.count_input(idx).to(x.dtype)
        if self.stream_dtype is not None:
            x = x.to(self.stream_dtype)
        x = self.transformer.drop(x)
        x0 = x
        documents = bounds = segments = None
        if self.config.document_bos >= 0:
            documents = document_ids(idx, self.config.document_bos)
            per_row = SEGMENTS_PER_ROW
            if self.config.smallest_window:
                segments = window_segments(documents, self.attention_window)
                per_row += self.config.block_size // self.config.smallest_window
            if idx.is_cuda and varlen_attn is not None:
                bounds = segment_bounds(documents if segments is None else segments, per_row)
        pairs = triples = None
        if self.config.bigram_table:
            pairs = self.bigram_table(bigram_rows(idx, self.bigram_table.num_embeddings, documents)).to(x.dtype)
        if self.config.trigram_table:
            triples = self.trigram_table(trigram_rows(idx, self.trigram_table.num_embeddings, documents)).to(x.dtype)
        dense = self.config.dense_connections
        kept = {source for sources in self.dense_sources.values() for source in sources} if dense else set()
        outputs = {-1: x0}
        value_mixture = None
        backout_stream = None
        dispersing = self.training and targets is not None and self.config.dispersion > 0
        sampled = []
        for layer, block in enumerate(self.transformer.h):
            if self.config.residual_weights:
                x = self.resid_weights[layer] * x + self.x0_weights[layer] * x0
            if pairs is not None:
                x = x + self.bigram_weights[layer] * pairs
            if triples is not None:
                x = x + self.trigram_weights[layer] * triples
            table = self.value_embeds[str(layer)] if str(layer) in self.value_embeds else None
            stem = self.stem_tables[str(layer)] if str(layer) in self.stem_tables else None
            x = block(x, table(idx).to(x.dtype) if table is not None else None, value_mixture, documents, bounds,
                      stem(idx) if stem is not None else None, segments)
            value_mixture = None
            if layer in kept:
                outputs[layer] = x
            if dense and layer in self.dense_sources:
                x, value_mixture = self.mix_earlier_outputs(layer, x, outputs)
            if self.config.backout and layer == backout_layer(self.config.n_layer):
                backout_stream = x
            if dispersing:
                sampled.append(x[:, DISPERSION_STRIDE - 1::DISPERSION_STRIDE])
        if backout_stream is not None:
            # the context that later layers needed but the prediction does not, as in modded-nanogpt's record 40
            x = x - self.backout_weight.to(x.dtype) * backout_stream
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.cap_logits(self.lm_head(x))
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            if dispersing:
                loss = loss + self.config.dispersion * dispersion_loss(torch.stack(sampled))
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.cap_logits(self.lm_head(x[:, [-1], :])) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        if not self.config.rotary:
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
