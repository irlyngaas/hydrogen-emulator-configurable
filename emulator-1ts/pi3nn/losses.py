"""
Masked-loss mechanism for PI3NN's up/down training on spatial data.

Unlike the flat PI3NN port (UQnet/pi3nn_torch/trainer.py), which splits
*rows* into disjoint up/down datasets before training ever starts, here
every pixel of every training example flows through net_mean, net_up,
AND net_down every iteration -- the up/down split is a per-pixel-
per-channel boolean mask (this pixel's mean-residual is positive or
negative) applied inside a masked-MSE loss, not a dataset split.

This creates a distributed-training subtlety the flat port's DDP
support doesn't have. The flat port's _local_shard keeps every rank's
shard the same *size*, which is sufficient there because up/down are
disjoint datasets split *before* sharding -- DDP's uniform gradient
averaging across ranks is then exactly correct. Here, masking happens
*after* sharding: two ranks with equal-sized pixel shards can have very
different *mask counts* (one rank's patches might skew toward positive
residuals). Naively computing local_sq_sum/local_mask_count per rank and
calling .backward() would let DDP's uniform per-rank averaging silently
bias the gradient away from the true global masked-MSE gradient.
"""
import torch
import torch.distributed as dist


def masked_sq_sum(pred, target, mask):
    """Returns (summed squared error over masked entries, mask count) as
    UNREDUCED tensors. Callers must combine these across batches/ranks
    before dividing -- never average a sequence of already-divided
    per-batch/per-rank means, which would misweight batches or ranks
    with different mask counts."""
    mask_f = mask.to(pred.dtype)
    sq_err = (pred - target) ** 2 * mask_f
    return sq_err.sum(), mask_f.sum()


def masked_mse_loss(pred, target, mask):
    """Single-process (or already-correctly-reduced) masked MSE. For the
    distributed case, use reduced_masked_mse_loss instead -- see its
    docstring for why this function alone isn't DDP-safe when mask
    counts can differ across ranks."""
    sq_sum, count = masked_sq_sum(pred, target, mask)
    return sq_sum / count.clamp_min(1)


def reduced_masked_mse_loss(pred, target, mask):
    """
    DDP-safe masked MSE: computes this rank's local sq_sum/mask count,
    all_reduces (SUM, no_grad) only the mask count across ranks to get
    the true global mask count, then returns

        world_size * local_sq_sum / global_mask_count

    The world_size factor exactly cancels DDP's gradient-averaging-by-
    world_size, so the synchronized gradient after .backward() equals
    d(global_sq_sum)/d(params) / global_mask_count -- the true global
    masked-MSE gradient -- regardless of how unevenly mask counts are
    distributed across ranks' local shards.

    When dist isn't initialized (single-process), this reduces exactly
    to masked_mse_loss. For the mean-training phase (mask = all-ones,
    every rank's shard the same size), this also reduces to today's
    ordinary uniform-averaging behavior -- one function covers all three
    PI3NN phases.
    """
    local_sq_sum, local_count = masked_sq_sum(pred, target, mask)
    if not dist.is_initialized():
        return local_sq_sum / local_count.clamp_min(1)

    global_count = local_count.detach().clone()
    dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
    world_size = dist.get_world_size()
    return world_size * local_sq_sum / global_count.clamp_min(1)


def residual_targets(net_mean, state, evaptrans, params, y):
    """net_mean must already be frozen (eval mode); this always calls it
    under no_grad regardless, since the mean-phase forward graph should
    never be touched while training up/down.

    Returns (up_target, up_mask, down_target, down_mask). Masked-out
    entries in up_target/down_target are arbitrary (clamped to 0) --
    they're excluded by the mask in the loss, their value never
    contributes."""
    with torch.no_grad():
        mean_pred = net_mean(state, evaptrans, params)
    diff = y - mean_pred
    up_mask = diff > 0
    down_mask = diff < 0
    up_target = diff.clamp(min=0)
    down_target = (-diff).clamp(min=0)
    return up_target, up_mask, down_target, down_mask
