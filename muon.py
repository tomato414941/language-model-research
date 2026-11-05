"""
Muon for the matrices of the hidden layers and AdamW for every other weight.

Adapted from MuonAdamW in karpathy/nanochat (nanochat/optim.py, MIT License, Copyright (c) 2025
Andrej Karpathy), which comes from KellerJordan/modded-nanogpt. This version runs in one process
and steps one weight at a time. Polar Express, NorMuon and cautious weight decay can each be turned
off, so that a change in the results can be traced to one of them.
"""

import torch

# The (a, b, c) of the iteration X = aX + b(XX^T)X + c(XX^T)^2X, which moves singular values to one.
NEWTON_SCHULZ = [(3.4445, -4.7750, 2.0315)] * 5
# From arXiv 2505.16932, computed for five iterations and a safety factor of 0.02.
POLAR_EXPRESS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

# modded-nanogpt's record 81 gives MUDD's weights a quarter of the rate its other AdamW weights take
DENSE_RATE_SHARE = 0.25

WARMDOWN_SHARE = 0.5
MOMENTUM_START = 0.85
MOMENTUM_END = 0.95
MOMENTUM_RISE_UPDATES = 300


def orthogonalize(matrices, coefficients):
    """Moves the singular values of each matrix in a batch towards one, in bfloat16."""
    x = matrices.bfloat16()
    x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    tall = x.size(-2) > x.size(-1)
    for a, b, c in coefficients:
        inner = x.mT @ x if tall else x @ x.mT
        polynomial = b * inner + c * (inner @ inner)
        x = a * x + (x @ polynomial if tall else polynomial @ x)
    return x.float()


def normalize_neurons(update, second_moment, beta2):
    """NorMuon: evens out the rows or columns of each matrix and keeps the size of the matrix."""
    reduced = -1 if update.size(-2) >= update.size(-1) else -2
    mean_square = update.square().mean(dim=reduced, keepdim=True)
    second_moment.lerp_(mean_square, 1 - beta2)
    scale = second_moment.clamp_min(1e-10).rsqrt()
    norm = mean_square.sum(dim=(-2, -1), keepdim=True).sqrt()
    scaled_norm = (mean_square * scale.square()).sum(dim=(-2, -1), keepdim=True).sqrt()
    return update * scale * (norm / scaled_norm.clamp_min(1e-10))


class MuonAdamW(torch.optim.Optimizer):
    """Steps each parameter group with Muon or AdamW, as the group's `kind` says."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            step = self._step_muon if group['kind'] == 'muon' else self._step_adamw
            for p in group['params']:
                if p.grad is not None:
                    step(p, group)

    def _step_adamw(self, p, group):
        state = self.state[p]
        if not state:
            state['step'] = 0
            state['average'] = torch.zeros_like(p)
            state['square'] = torch.zeros_like(p)
        state['step'] += 1
        beta1, beta2 = group['betas']
        state['average'].lerp_(p.grad, 1 - beta1)
        state['square'].lerp_(p.grad.square(), 1 - beta2)
        square = state['square'] / (1 - beta2 ** state['step'])
        step_size = group['lr'] / (1 - beta1 ** state['step'])
        average = state['average']
        if group.get('cautious'):
            # Cautious AdamW (arXiv 2411.16085): no step where the average points against the gradient, and the
            # other steps scaled up to keep the size, as in its authors' code
            agrees = (average * p.grad > 0).to(average.dtype)
            average = average * agrees * (agrees.numel() / (agrees.sum() + 1))
        p.addcdiv_(average, square.sqrt() + group['eps'], value=-step_size)

    def _step_muon(self, p, group):
        state = self.state[p]
        if not state:
            state['momentum'] = torch.zeros_like(p)
        momentum = group['momentum']
        state['momentum'].lerp_(p.grad, 1 - momentum)
        nesterov = p.grad.lerp(state['momentum'], momentum)

        # A matrix that holds several projections is treated as that many matrices.
        matrices = torch.stack(nesterov.chunk(group['chunks'], dim=0))
        coefficients = POLAR_EXPRESS if group['polar_express'] else NEWTON_SCHULZ
        update = orthogonalize(matrices, coefficients)
        rows, columns = update.shape[-2:]
        if group['normalize_neurons']:
            if 'second_moment' not in state:
                shape = (len(update), rows, 1) if rows >= columns else (len(update), 1, columns)
                state['second_moment'] = torch.zeros(shape, dtype=update.dtype, device=update.device)
            update = normalize_neurons(update, state['second_moment'], group['beta2'])
        update = update.flatten(0, 1)

        decay = group['weight_decay'] * p
        if group['cautious_decay']:
            decay = decay * (update * p >= 0)
        lr = group['lr'] * max(1.0, rows / columns) ** 0.5
        p.sub_(lr * (update + decay))


def configure_muon(model, muon_lr, other_lr, value_embedding_lr=0.1, polar_express=True,
                   neuron_normalization=True, cautious_decay=True, betas=(0.8, 0.95), eps=1e-10, beta2=0.95,
                   embedding_lr=0.6, scalar_lr=0.5, cautious_adamw=False):
    """Gives the matrices of the blocks to Muon and every other weight to AdamW, cautious if cautious_adamw.

    A token table that only reads gets embedding_lr. A table shared with the predictions gets other_lr.
    The x0 weights and the COUNT gate get scalar_lr and the resid weights a hundredth of it, since they
    multiply through the layers, as in karpathy/nanochat. The table of token pairs takes the value embeddings'
    rate and its weights, like the backout's, the x0 weights' rate, as their counterparts in modded-nanogpt do.
    The table of triples and STEM's tables, which start at zero like the table of pairs, take what it takes.
    MUDD's weights get a share of other_lr with a slower second moment, as in modded-nanogpt's record 81.
    """
    matrices, query_key_value, value_embeddings, reading_table, resid_weights, x0_weights, others = ([] for _ in range(7))
    pair_table, dense = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith('value_embeds.'):
            value_embeddings.append(p)
        elif name == 'transformer.wte.weight' and model.config.separate_embeddings:
            reading_table.append(p)
        elif name in ('bigram_table.weight', 'trigram_table.weight') or name.startswith('stem_tables.'):
            pair_table.append(p)
        elif name == 'resid_weights':
            resid_weights.append(p)
        elif name in ('x0_weights', 'count_gate', 'bigram_weights', 'trigram_weights', 'backout_weight'):
            x0_weights.append(p)
        elif name.startswith('dense_'):
            dense.append(p)
        elif not name.startswith('transformer.h.') or p.dim() != 2:
            others.append(p)
        elif name.endswith('c_attn.weight'):
            query_key_value.append(p)
        else:
            matrices.append(p)
    muon = dict(kind='muon', lr=muon_lr, momentum=MOMENTUM_START, weight_decay=0.0, beta2=beta2,
                polar_express=polar_express, normalize_neurons=neuron_normalization,
                cautious_decay=cautious_decay)
    adamw = dict(kind='adamw', betas=betas, eps=eps, cautious=cautious_adamw)
    groups = [
        dict(adamw, params=others, lr=other_lr, initial_lr=other_lr),
        dict(muon, params=matrices, chunks=1, initial_lr=muon_lr),
        dict(muon, params=query_key_value, chunks=3, initial_lr=muon_lr),
        dict(adamw, params=value_embeddings, lr=value_embedding_lr, initial_lr=value_embedding_lr),
        dict(adamw, params=reading_table, lr=embedding_lr, initial_lr=embedding_lr),
        dict(adamw, params=resid_weights, lr=scalar_lr * 0.01, initial_lr=scalar_lr * 0.01),
        dict(adamw, params=x0_weights, lr=scalar_lr, initial_lr=scalar_lr, betas=(0.96, betas[1])),
        dict(adamw, params=pair_table, lr=value_embedding_lr, initial_lr=value_embedding_lr),
        dict(adamw, params=dense, lr=other_lr * DENSE_RATE_SHARE, initial_lr=other_lr * DENSE_RATE_SHARE,
             betas=(0.9, 0.99)),
    ]
    return MuonAdamW([group for group in groups if group['params']])


def set_schedule(optimizer, update, total_updates, weight_decay, warmdown_share=WARMDOWN_SHARE):
    """Sets the values for this update of a run of total_updates and returns the share of the full learning rate in
    use, which falls in a straight line to zero over the last `warmdown_share` of training."""
    return set_schedule_at(optimizer, update, min(update / total_updates, 1.0), weight_decay, warmdown_share)


def set_schedule_at(optimizer, update, progress, weight_decay, warmdown_share=WARMDOWN_SHARE):
    """Sets the values for this update, `progress` of the way through training, as set_schedule does. A run with a
    time budget passes the share of its time spent."""
    progress = min(progress, 1.0)
    share = min((1 - progress) / warmdown_share, 1.0)
    rise = min(update / MOMENTUM_RISE_UPDATES, 1.0)
    momentum = MOMENTUM_START + rise * (MOMENTUM_END - MOMENTUM_START)
    for group in optimizer.param_groups:
        group['lr'] = group['initial_lr'] * share
        if group['kind'] == 'muon':
            group['momentum'] = momentum
            group['weight_decay'] = weight_decay * (1 - progress)
    return share
