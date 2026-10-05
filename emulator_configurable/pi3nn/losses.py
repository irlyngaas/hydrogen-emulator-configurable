"""
Masked-loss mechanism for PI3NN's up/down training, vendored (with one
adaptation, see residual_targets below) from emulator-1ts/pi3nn/losses.py.

Every pixel of every training example flows through net_mean, net_up,
AND net_down every iteration -- the up/down split is a per-pixel
boolean mask (this pixel's mean-residual is positive or negative)
applied inside a masked-MSE loss, not a dataset split.

This creates a distributed-training subtlety: masking happens AFTER
DDP sharding, so two ranks' local shards can have very different mask
counts (one rank's patches might skew toward positive residuals).
Naively computing local_sq_sum/local_mask_count per rank and calling
.backward() would let DDP's uniform per-rank gradient averaging
silently bias the gradient away from the true global masked-MSE
gradient. This applies identically whether DDP is hand-rolled
(emulator-1ts) or Lightning-managed (here) -- Lightning's
strategy='ddp' is its own wrapper around the same underlying
torch.nn.parallel.DistributedDataParallel, with the same gradient-
averaging-by-world-size hook firing after PI3NNUpDownModule's
training_step loss.backward() (called internally by Lightning's
automatic-optimization loop), so the fix below is equally necessary
here.
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

    When dist isn't initialized (single-process/single-GPU, no
    strategy='ddp'), this reduces exactly to masked_mse_loss. For the
    mean-training phase (mask = all-ones, every rank's shard the same
    size), this also reduces to ordinary uniform-averaging behavior --
    one function covers all three PI3NN phases.
    """
    local_sq_sum, local_count = masked_sq_sum(pred, target, mask)
    if not dist.is_initialized():
        return local_sq_sum / local_count.clamp_min(1)

    global_count = local_count.detach().clone()
    dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
    world_size = dist.get_world_size()
    return world_size * local_sq_sum / global_count.clamp_min(1)


def residual_targets(target, mean_pred):
    """Adapted, not copied verbatim, from emulator-1ts's version: that one
    takes net_mean itself and calls it internally. Here the caller
    (networks.predict_updown) already computed mean_pred once -- for the
    recurrent_clone variant especially, calling net_mean a second time
    would mean a wasted full autoregressive rollout, so this function
    just takes the already-computed tensors.

    Returns (up_target, up_mask, down_target, down_mask). Masked-out
    entries in up_target/down_target are arbitrary (clamped to 0) --
    they're excluded by the mask in the loss, their value never
    contributes. Purely elementwise, so the extra timestep axis in
    this package's (batch, T, C, H, W) tensors (vs. emulator-1ts's
    (batch, C, H, W)) needs no changes here."""
    diff = target - mean_pred
    up_mask = diff > 0
    down_mask = diff < 0
    up_target = diff.clamp(min=0)
    down_target = (-diff).clamp(min=0)
    return up_target, up_mask, down_target, down_mask
