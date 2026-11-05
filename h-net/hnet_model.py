"""H-Net (Hwang, Wang and Gu, arXiv 2507.07955): bytes in, chunks that the model learns to draw, model.py's GPT
blocks over the chunks, and bytes out.

One stage, as the paper's 1-stage H-Net. An encoder runs over the bytes. A routing module marks where a chunk
begins, by how far each byte's vector turns from the one before it. The main network runs over the vectors at those
marks. A dechunking layer smooths the main network's outputs by the marks' probabilities and gives each byte the
output of its chunk, and a decoder runs over the bytes and predicts the next one. These parts follow the authors'
code (github.com/goombalab/hnet, MIT): the routing, the smoothing, the residual projection that starts at zero, the
straight-through confidence, and the ratio loss with its weight of 0.03 and its target of 6 bytes per chunk. While
chunks are finer than the target, the ratio loss's weight rises above 0.03, so that the main network's cost stays
near the target's; the paper keeps it at 0.03.

The encoder and decoder layers are Mamba-2 as in the paper, or Mamba-3 or Gated DeltaNet in their place; 'conv' is a
causal convolution that runs without a GPU, for tests. The main network is the second bundle's: model.py's blocks
with its norms, rotary positions and the x0 and resid weights. The chunks of all rows are packed into one sequence
that attends only within each row, so that a row with fewer chunks costs less. Finding the chunks is the one point
in a pass that waits for the GPU. On a GPU the main network compiles as one graph for any number of chunks, and the
steps between the outer layers' mixers, the routing and the prediction compile too.

train.py trains it when a config names HNet as GPT and HNetConfig, with these fields set, as GPTConfig; see
h-net/configs.
"""

import importlib.util
import inspect
import math
import sys
import types
from dataclasses import dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import GPT, Block, GPTConfig, RMSNorm, make_norm

try:
    from torch.nn.attention.varlen import varlen_attn
except ImportError:
    varlen_attn = None

BOS = 256  # the token that begins a document, after the 256 byte values
SMOOTHING_BLOCK = 64
REPORT_EVERY = 100  # training passes between reports of the bytes per chunk


@dataclass
class HNetConfig(GPTConfig):
    """GPTConfig's fields describe the main network, except block_size, which counts the bytes of a row."""

    outer: str = "mamba2"  # the encoder and decoder layers: 'mamba2', 'mamba3', 'gated_deltanet' or 'conv'
    outer_width: int = 512
    encoder_layers: int = 4
    decoder_layers: int = 4
    target_ratio: float = 6.0  # the bytes per chunk that the ratio loss aims for
    ratio_weight: float = 0.03  # the ratio loss's weight at the start, and its least
    # How fast the log of the ratio loss's weight moves in training, per 512 rows and per unit of target_ratio times
    # the share of bytes that begin a chunk, less one: up while chunks are finer than the target, and back down to
    # ratio_weight while they are coarser. At most 1. Zero keeps the weight at ratio_weight, as the paper does.
    ratio_weight_rate: float = 0.03
    compile_main: bool = True  # on a GPU, compile the main network, the steps around the mixers and the prediction
    # Above zero, a chunk begins at every this many bytes, whatever the routing says. Only for timing: early in
    # training the routing begins a chunk at about every other byte, which a short timing would take as the cost.
    fixed_chunks: int = 0


class FirstOutput(nn.Module):
    """A layer that returns a tuple, of which only the first item is kept."""

    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        return self.layer(x)[0]


class CausalConvolution(nn.Module):
    """A gated causal convolution over the last four bytes. It runs anywhere, so tests use it."""

    def __init__(self, width, size=4):
        super().__init__()
        self.in_proj = nn.Linear(width, 2 * width, bias=False)
        self.conv = nn.Conv1d(width, width, size, groups=width, padding=size - 1)
        self.out_proj = nn.Linear(width, width, bias=False)

    def forward(self, x):
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        value = self.conv(value.transpose(1, 2))[..., : x.size(1)].transpose(1, 2)
        return self.out_proj(value * F.silu(gate))


def without_package_imports(package):
    """Let `package`'s modules be imported without running its __init__.py. mamba_ssm's loads Mamba-1's CUDA
    extension and Mamba-3's TileLang and CuTe kernels, none of which Mamba-2's Triton kernels need."""
    if package not in sys.modules:
        spec = importlib.util.find_spec(package)
        module = types.ModuleType(package)
        module.__path__ = list(spec.submodule_search_locations)
        sys.modules[package] = module


def outer_mixer(kind, width, index):
    """A layer that mixes the bytes along a row and has no MLP; the three real ones hold about 6 width² weights."""
    if kind == "mamba2":
        without_package_imports("mamba_ssm")
        from causal_conv import install
        from mamba_ssm.modules.mamba2 import Mamba2

        # The paper's encoder and decoder layers, as in its configs/hnet_1stage_L.json, on Mamba-2's fused path,
        # which recomputes its convolution in the backward pass. causal-conv1d's CUDA functions, which that path
        # calls, have no build for this PyTorch, so the convolution is causal_conv.py's.
        install()
        return Mamba2(d_model=width, d_state=128, d_conv=4, expand=2, headdim=64, chunk_size=256, layer_idx=index)
    if kind == "mamba3":
        from mamba_ssm import Mamba3

        return Mamba3(d_model=width, d_state=128, expand=2, headdim=64, chunk_size=64, layer_idx=index)
    if kind == "gated_deltanet":
        from fla.layers import GatedDeltaNet

        # fla asks for heads of 0.75 width in all with its output gate, for the size of a Mamba-2 layer.
        heads = width * 3 // 4 // 128
        return FirstOutput(GatedDeltaNet(hidden_size=width, expand_v=2, head_dim=128, num_heads=heads, mode="chunk",
                                         use_gate=True, use_short_conv=True, layer_idx=index))
    if kind == "conv":
        return CausalConvolution(width)
    raise ValueError(f"unknown outer layer: {kind}")


def normalized(x, dtype):
    """An outer layer's pre-norm, in the type its mixer reads if one is given, as autocast would give it."""
    x = F.rms_norm(x, (x.size(-1),))
    return x if dtype is None else x.to(dtype)


def added(x, y):
    return x + y.float()


class OuterLayer(nn.Module):
    """A pre-norm residual layer over the bytes, its stream kept in float32 as the authors' blocks keep it. HNet sets
    input_dtype when it trains in bfloat16, and on a GPU gives it compiled versions of the steps around the mixer."""

    def __init__(self, kind, width, index):
        super().__init__()
        self.mixer = outer_mixer(kind, width, index)
        self.input_dtype = None
        self.normalize, self.add = normalized, added

    def forward(self, x):
        return self.add(x, self.mixer(self.normalize(x, self.input_dtype)))


class STE(torch.autograd.Function):
    """Ones forward, and the gradient passed through unchanged backward."""

    @staticmethod
    def forward(ctx, x):
        return torch.ones_like(x)

    @staticmethod
    def backward(ctx, grad):
        return grad


def turn(x, cos, sin):
    """model.rotate for packed vectors: x of shape (tokens, heads, head size), cos and sin of (tokens, 1, half)."""
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    half = x.size(-1) // 2
    first, second = x[..., :half], x[..., half:]
    return torch.cat([first * cos + second * sin, second * cos - first * sin], dim=-1)


def causal_within_rows(q, k, v, starts, longest):
    """Causal attention within each row of a packed sequence; q, k and v are (tokens, heads, head size), starts
    holds each row's first token and, last, their number, and no row is longer than `longest`."""
    if q.is_cuda:
        # FlashAttention's variable-length kernel; (-1, 0) is the causal window.
        out = varlen_attn(q.bfloat16(), k.bfloat16(), v.bfloat16(), starts, starts, longest, longest,
                          window_size=(-1, 0))
        return out.to(q.dtype)
    rows = []
    for begin, end in zip(starts[:-1].tolist(), starts[1:].tolist()):
        row = [t[begin:end].transpose(0, 1)[None] for t in (q, k, v)]
        rows.append(F.scaled_dot_product_attention(*row, is_causal=True)[0].transpose(0, 1))
    return torch.cat(rows)


def segment_sums(x):
    """For x of shape (..., L), the (..., L, L) sums of x over (j, i] where j <= i, and -inf where j > i. Sums are
    taken directly rather than as differences, so that a -inf in x gives -inf and never a NaN."""
    length = x.size(-1)
    lower = torch.tril(torch.ones(length, length, dtype=torch.bool, device=x.device), diagonal=-1)
    sums = x[..., :, None].expand(*x.shape, length).masked_fill(~lower, 0).cumsum(dim=-2)
    return sums.masked_fill(~torch.tril(torch.ones_like(lower), diagonal=0), -torch.inf)


def smooth(values, probability, fresh):
    """The paper's smoothing of the chunks' outputs, z_c = p_c v_c + (1 - p_c) z_{c-1}, along the packed chunks,
    starting afresh where `fresh` is set: at each row's first chunk. In float32, within blocks by a matrix of decays
    and across blocks by the same over the blocks' last values."""
    count, width = values.shape
    p = probability.float().clamp(1e-4, 1 - 1e-4)  # as the authors clamp it
    inputs = p[:, None] * values.float()
    decay = torch.where(fresh, -torch.inf, torch.log1p(-p))
    pad = (-count) % SMOOTHING_BLOCK
    inputs = F.pad(inputs, (0, 0, 0, pad)).view(-1, SMOOTHING_BLOCK, width)
    decay = F.pad(decay, (0, pad), value=-torch.inf).view(-1, SMOOTHING_BLOCK)
    within = torch.exp(segment_sums(decay)) @ inputs  # each block as if it started from zero
    from_start = torch.exp(decay.cumsum(dim=-1))  # what is left of the state entering a block, at each position
    # The state at each block's end, z_k = within_k + D_k z_{k-1}, solved the same way over the blocks.
    ends = torch.exp(segment_sums(decay.sum(dim=-1))) @ within[:, -1]
    entering = F.pad(ends[:-1], (0, 0, 1, 0))
    out = within + from_start[..., None] * entering[:, None, :]
    return out.view(-1, width)[:count]


class HNet(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert not config.count and not config.value_embeddings, "the main network reads chunks, not tokens"
        assert config.n_embd >= config.outer_width
        self.config = config
        width = config.outer_width
        main = replace(config, value_embeddings=False, dispersion=0.0)
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, width),
            h=nn.ModuleList([Block(main, layer) for layer in range(config.n_layer)]),
            ln_f=make_norm(main),
        ))
        if config.residual_weights:
            self.resid_weights = nn.Parameter(torch.ones(config.n_layer))
            self.x0_weights = nn.Parameter(torch.full((config.n_layer,), 0.1))
        outer = config.encoder_layers + config.decoder_layers
        self.encoder = nn.ModuleList([OuterLayer(config.outer, width, i) for i in range(config.encoder_layers)])
        self.decoder = nn.ModuleList([OuterLayer(config.outer, width, config.encoder_layers + i)
                                      for i in range(config.decoder_layers)])
        self.routing_query = nn.Linear(width, width, bias=False)
        self.routing_key = nn.Linear(width, width, bias=False)
        # The chunks' vectors are widened to the main network's width by a learned tail, as the authors do.
        self.widening = nn.Parameter(torch.zeros(config.n_embd - width)) if config.n_embd > width else None
        self.residual = nn.Linear(width, width)
        self.lm_head = nn.Linear(width, config.vocab_size, bias=False)
        self.input_norm = RMSNorm() if config.separate_embeddings else None
        self.stream_dtype = None
        self.passes = 0
        self.selected = None  # the share of bytes that began a chunk in the last training pass, kept on its device
        floor = math.log(config.ratio_weight) if config.ratio_weight > 0 else -math.inf
        self.register_buffer("ratio_log_weight", torch.tensor(floor), persistent=False)
        if not config.separate_embeddings:
            self.transformer.wte.weight = self.lm_head.weight

        # The main network starts as model.GPT starts.
        self.transformer.h.apply(self._init_main)
        for name, p in self.transformer.h.named_parameters():
            if name.endswith("c_proj.weight"):
                if config.zero_output:
                    torch.nn.init.zeros_(p)
                else:
                    torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        # The outer layers' linear weights start as the authors' do; their other weights as their libraries set them.
        for layer in [*self.encoder, *self.decoder]:
            for name, module in layer.named_modules():
                if isinstance(module, nn.Linear):
                    last = name.endswith(("out_proj", "o_proj"))
                    torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(outer) if last else 0.02)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
        with torch.no_grad():
            self.routing_query.weight.copy_(torch.eye(width))
            self.routing_key.weight.copy_(torch.eye(width))
        torch.nn.init.zeros_(self.residual.weight)
        if config.separate_embeddings:
            torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
            torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        else:
            torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        self.route, self.main_network, self.predict = self._route, self._main_network, self._predict
        if config.compile_main and torch.cuda.is_available():
            # The main network is one graph whatever the number of chunks, since the attention kernel is told the
            # longest a row can be rather than the longest that this pass holds. The other parts keep their shapes.
            self.route = torch.compile(self._route)
            self.main_network = torch.compile(self._main_network, dynamic=True)
            self.predict = torch.compile(self._predict)
            steps = torch.compile(normalized), torch.compile(added)
            for layer in (*self.encoder, *self.decoder):
                layer.normalize, layer.add = steps
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    @property
    def bytes_per_chunk(self):
        """Read only when reported, since reading it waits for the GPU."""
        return float("nan") if self.selected is None else 1 / max(self.selected.item(), 1e-9)

    @staticmethod
    def _init_main(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    # The calls that train.py and loading.py make of a model.
    def use_flash_attention(self, function):
        """Packed attention takes PyTorch's variable-length kernel, so another function is not used."""

    def use_count_table(self, table):
        raise ValueError("H-Net reads bytes, which a COUNT table does not count")

    def keep_stream_in(self, dtype):
        """The main network's stream, and what the outer layers' mixers read; the outer stream stays in float32."""
        self.stream_dtype = dtype
        for layer in (*self.encoder, *self.decoder):
            layer.input_dtype = dtype

    def crop_block_size(self, block_size):
        raise ValueError("train it with the block size it was made for")

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and self.config.separate_embeddings:
            n_params -= self.transformer.wte.weight.numel()
        return n_params

    configure_optimizers = GPT.configure_optimizers

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """Model FLOPs utilization against an A100's 312 TFLOPS, as model.GPT reports it, counting the main network
        once per chunk at the last measured bytes per chunk."""
        main = sum(p.numel() for p in self.transformer.h.parameters())
        outer = self.get_num_params() - main
        ratio = self.bytes_per_chunk if math.isfinite(self.bytes_per_chunk) else self.config.target_ratio
        flops = 6 * (outer + main / ratio) * self.config.block_size * fwdbwd_per_iter
        return flops / dt / 312e12

    def _route(self, x):
        """The encoder's normalized output, and each byte's probability of beginning a chunk, from the angle between
        its vector and the one before it, and whether it begins one. A row's first byte always begins one."""
        encoded = F.rms_norm(x, (x.size(-1),))
        cosine = F.cosine_similarity(self.routing_query(encoded[:, :-1]).float(),
                                     self.routing_key(encoded[:, 1:]).float(), dim=-1)
        probability = F.pad(((1 - cosine) / 2).clamp(0.0, 1.0), (1, 0), value=1.0)
        return encoded, probability, probability > 0.5

    def _main_network(self, x, positions, starts, longest):
        """model.GPT's blocks with model.CausalSelfAttention's weights and steps, on packed chunks."""
        cos = sin = None
        if self.config.rotary:
            attn = self.transformer.h[0].attn
            cos, sin = attn.rotary_cos[0, positions], attn.rotary_sin[0, positions]
        x0 = x
        for layer, block in enumerate(self.transformer.h):
            if self.config.residual_weights:
                x = self.resid_weights[layer] * x + self.x0_weights[layer] * x0
            q, k, v = self._project(block, x, cos, sin)
            x = self._finish(block, x, causal_within_rows(q, k, v, starts, longest))
        return self.transformer.ln_f(x)

    @staticmethod
    def _project(block, x, cos, sin):
        attn = block.attn
        count, width = x.shape
        size = width // attn.n_head
        q, k, v = (t.view(count, attn.n_head, size) for t in attn.c_attn(block.ln_1(x)).split(width, dim=1))
        if cos is not None:
            q, k = turn(q, cos, sin), turn(k, cos, sin)
        if attn.qk_norm is not None:
            q, k = attn.qk_norm(q), attn.qk_norm(k)
        return q, k, v

    @staticmethod
    def _finish(block, x, y):
        x = x + block.attn.resid_dropout(block.attn.c_proj(y.reshape(x.shape)))
        return x + block.mlp(block.ln_2(x))

    def _predict(self, x, targets=None):
        """The logits of every position, and their loss against the targets if there are any."""
        logits = self.lm_head(F.rms_norm(x, (x.size(-1),)))
        if self.config.logit_softcap > 0:
            cap = self.config.logit_softcap
            logits = cap * torch.tanh(logits.float() / cap)
        if targets is None:
            return logits, None
        return logits, F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)

    def adjust_ratio_weight(self, selected, rows):
        """Moves the ratio loss's weight by how far this pass's share of chunk starts was from the target's, on the
        GPU, without waiting for it."""
        rate = self.config.ratio_weight_rate * rows / 512
        if rate and self.config.ratio_weight > 0:
            moved = self.ratio_log_weight + rate * (self.config.target_ratio * selected - 1)
            self.ratio_log_weight.copy_(moved.clamp(math.log(self.config.ratio_weight), 0.0))

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.config.block_size, f"rows hold at most {self.config.block_size} bytes"
        x = self.transformer.wte(idx).float()
        if self.input_norm is not None:
            x = self.input_norm(x)
        for layer in self.encoder:
            x = layer(x)

        encoded, probability, boundary = self.route(x)
        if self.config.fixed_chunks:
            boundary = (torch.arange(t, device=idx.device) % self.config.fixed_chunks == 0).expand(b, t)
        # The chunks of every row, one row after another. Their number sizes what follows, so finding them waits
        # for the GPU; nothing else in the pass does.
        counts = boundary.sum(dim=1)
        starts = F.pad(counts.cumsum(0), (1, 0)).to(torch.int32)
        marks = boundary.reshape(-1).nonzero().squeeze(1)
        within = boundary.cumsum(dim=1) - 1  # each byte's chunk, counted within its row
        positions = within.reshape(-1)[marks]
        chunks = encoded.reshape(b * t, -1)[marks]
        if self.widening is not None:
            chunks = torch.cat([chunks, self.widening.to(chunks.dtype).expand(len(chunks), -1)], dim=-1)
        if self.stream_dtype is not None:
            chunks = chunks.to(self.stream_dtype)
        # A row holds at most t chunks.
        out = self.main_network(chunks, positions, starts, t)[:, : self.config.outer_width]

        smoothed = smooth(out, probability.reshape(-1)[marks], positions == 0)
        spread = smoothed[(starts[:-1, None] + within).reshape(-1)].view(b, t, -1)
        confidence = torch.where(boundary, probability, 1 - probability)
        with torch.autocast(device_type=idx.device.type, enabled=False):
            residual = self.residual(encoded.float())
        x = spread * STE.apply(confidence)[..., None] + residual
        for layer in self.decoder:
            x = layer(x)

        if targets is None:
            return self._predict(x[:, [-1]])
        logits, loss = self.predict(x, targets)
        selected = boundary.float().mean()
        ratio = self.config.target_ratio
        ratio_loss = ratio / (ratio - 1) * ((ratio - 1) * selected * probability.mean()
                                            + (1 - selected) * (1 - probability.mean()))
        loss = loss + self.ratio_log_weight.exp() * ratio_loss
        if self.training:
            self.selected = selected.detach()
            self.adjust_ratio_weight(self.selected, b)
            self.passes += 1
            if self.passes % REPORT_EVERY == 0:
                print(f"h-net: {self.passes} passes, {self.bytes_per_chunk:.2f} bytes per chunk, ratio weight "
                      f"{self.ratio_log_weight.exp().item():.4f}", flush=True)
        return logits, loss


def byte_language_model(network, device, batch_tokens):
    """The model as loading.LanguageModel sees one: bytes for tokens, and BOS before each document. On a GPU it
    runs in bfloat16 as in training, its weights kept in float32 for the parts that need them."""
    from loading import LanguageModel

    def logits(tokens):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=tokens.is_cuda):
            return network(tokens, tokens)[0]

    return LanguageModel(
        logits=logits,
        encode=lambda text: list(text.encode("utf-8")),
        context=network.config.block_size,
        prefix=BOS,
        device=device,
        batch_tokens=batch_tokens,
    )


def load(directory, device):
    """A model that train.py saved in directory; h-net/configs/hnet.py saves the fields of HNetConfig with it."""
    checkpoint = torch.load(f"{directory}/ckpt.pt", map_location=device)
    fields = {**checkpoint["model_args"], "compile_main": False}
    known = set(inspect.signature(HNetConfig).parameters)
    network = HNet(HNetConfig(**{key: value for key, value in fields.items() if key in known}))
    network.load_state_dict({key.removeprefix("_orig_mod."): value for key, value in checkpoint["model"].items()})
    network.to(device)
    network.eval()
    return network
