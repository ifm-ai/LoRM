"""The causal language-modeling loss that training algorithms and evaluation share.

Labels are the unshifted input ids with -100 where no target counts (lorm/data/tokens.py); the loss
shifts once, so the logits at position t predict token t + 1.

The cross-entropy follows Barron et al., 2026 (arXiv:2610.31642). The standard gradient,
softmax - one_hot, rounds the target's entry to zero once the target's logit leads the others by
about 16.6 in float32, while the other entries stay positive, so the gradient no longer sums to
zero ("softmax collapse"). The loss keeps the standard value; its gradient comes from the
log-sum-exp of the other logits relative to the target's, which gives the target's entry directly.
"""

import torch
from torch.nn import functional as F


class _CrossEntropy(torch.autograd.Function):
    """Hard-target cross-entropy of float logits (rows, vocabulary), per row or summed as
    F.cross_entropy computes it; rows whose target is -100 give zero loss and zero gradient."""

    @staticmethod
    def forward(ctx, logits, targets, reduction):
        valid = targets != -100
        target = targets.clamp_min(0)[:, None]
        relative = logits - logits.gather(-1, target)
        relative.scatter_(-1, target, float("-inf"))
        others = torch.logsumexp(relative, -1)
        ctx.save_for_backward(relative, others, target, valid)
        return F.cross_entropy(logits, targets, reduction=reduction)

    @staticmethod
    def backward(ctx, grad):
        relative, others, target, valid = ctx.saved_tensors
        grad = torch.where(valid, grad.expand(valid.shape), 0.0)
        # The loss is softplus(others), so its gradient is exp(z_j - z_t - softplus(others)) for j != t, in one exponent
        # because exp(z_j - z_t) overflows past a gap of about 88.7 in float32. The target takes minus their sum,
        # sigmoid(others), which stays nonzero however large its lead.
        gradient = (relative - F.softplus(others)[:, None]).exp() * grad[:, None]
        gradient.scatter_(-1, target, (-grad * torch.sigmoid(others))[:, None])
        return gradient, None, None


def causal_loss(logits, labels, reduction="sum"):
    """Next-token cross-entropy, summed unless `reduction` is "none"; callers normalize by their targets."""
    return _CrossEntropy.apply(logits[:, :labels.shape[1] - 1].float().reshape(-1, logits.shape[-1]),
                               labels[:, 1:].reshape(-1), reduction)
