"""Terminal Shrinkage Averaging (arXiv 2609.25482): the last weights of training, mixed with the average of the
weights it had at a few updates evenly apart up to the last. The average holds the last weights too.

Training does not change. The mixed weights are a second model of the same run, which rented.sh scores beside
the last weights, so that the two differ only by the mixing.
"""

import torch


def averaged_updates(last, count, every):
    """The updates after which training keeps the weights it averages: `count` of them, `every` updates apart, the
    last of them after update `last`."""
    updates = {last - every * back for back in range(count)}
    if updates and min(updates) <= 0:
        raise ValueError("the weights to average must all come after the first update; average fewer or closer ones")
    return updates


class WeightAverage:
    """The average of the named tensors of the state dictionaries that it is given, kept in 32 bits."""

    def __init__(self, names):
        self.names = set(names)
        self.total = {}
        self.count = 0

    @torch.no_grad()
    def add(self, state):
        for name in self.names:
            value = state[name].to(torch.float32)
            if name in self.total:
                self.total[name] += value
            else:
                # A copy, since training goes on changing the tensors of its state in place.
                self.total[name] = value.clone()
        self.count += 1

    @torch.no_grad()
    def mixed(self, last, share):
        """`last` with each averaged tensor made of `share` of the average and the rest of itself, in its own type."""
        return {
            name: ((1 - share) * value.to(torch.float32) + share * self.total[name] / self.count).to(value.dtype)
            if name in self.names else value.clone()
            for name, value in last.items()
        }
