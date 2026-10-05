"""
Small-scale validation for the Conv2D PI3NN integration, mirroring
run_boston_comparison.py's role for the flat PyTorch PI3NN port: a
standalone, runnable correctness check before trusting this on real
Frontier-scale CONUS1 data. This repo has no pytest infrastructure
anywhere (checked), so this follows the existing convention of a plain
runnable script with assertions, not a new test framework.

Two modes:

  --mode full
      Single-process, tiny synthetic 2D dataset through the real
      PI3NNConvTrainer pipeline end-to-end (train -> boundary_optimization
      -> evaluate). Asserts net_up/net_down outputs are strictly positive
      everywhere (the whole point of PositiveResNetWrapper's activation),
      and that per-channel PICP on the training set lands close to the
      target quantile -- a real pass/fail check: BoundaryOptimizer's
      bisection search is *defined* to hit close to the target outlier
      count on the set it was calibrated against, regardless of how well
      net_mean/net_up/net_down are actually trained, so this mainly
      verifies the whole pipeline is wired together correctly (shapes,
      masking, per-channel calibration), not model quality.

  --mode ddp_loss
      Isolated correctness check of losses.reduced_masked_mse_loss
      specifically, not the full pipeline -- this is the one piece of
      DDP mechanics in this whole integration that's genuinely new
      relative to the already-Frontier-validated flat port (see
      losses.py's module docstring for why). Uses a real DDP-wrapped
      tiny shared model so DDP's automatic gradient-averaging hook
      actually fires, splits data into two ranks with DELIBERATELY
      uneven mask counts, and asserts the resulting (DDP-averaged)
      gradient matches a single-process reference computed on the full,
      unsharded data with the same (pre-update) weights. Run with
      exactly 2 processes -- via srun (reads SLURM's env vars, same as
      every other distributed launch in this project -- see
      run_pi3nn_validate.slurm) or via torchrun for ad hoc local testing
      without SLURM.

Usage (run from emulator-1ts/ as a module, same convention as the
validated flat port's `python3 -m pi3nn_torch.run_boston_comparison`):
  python3 -m pi3nn.validate_synthetic --mode full
  torchrun --nproc_per_node=2 --standalone -m pi3nn.validate_synthetic --mode ddp_loss
"""
import argparse
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

from .networks import build_networks
from .trainer import PI3NNConvTrainer
from .losses import reduced_masked_mse_loss


def custom_collate(batch):
    """Inlined copy of main.py's custom_collate rather than importing it --
    importing main.py transitively imports dataset.py, which needs
    xarray/xbatcher/parflow.tools.io (not needed here, and not installed
    in every environment this lightweight synthetic check should be able
    to run in)."""
    s, e, p, y = [], [], [], []
    for b in batch:
        s.append(b[0])
        e.append(b[1])
        p.append(b[2])
        y.append(b[3])
    return torch.stack(s), torch.stack(e), torch.stack(p), torch.stack(y)


class SyntheticDataset(Dataset):
    """Random tensors at the shapes ParFlowDataset.__getitem__ would
    produce: (state, evaptrans, params, target), no batch dim, channel
    counts matching make_model_def() below."""

    def __init__(self, n, state_shape, evaptrans_shape, params_shape, out_shape, seed, target_bias=0.0):
        g = torch.Generator().manual_seed(seed)
        self.state = torch.randn((n, *state_shape), generator=g, dtype=torch.float64)
        self.evaptrans = torch.randn((n, *evaptrans_shape), generator=g, dtype=torch.float64)
        self.params = torch.randn((n, *params_shape), generator=g, dtype=torch.float64)
        self.target = torch.randn((n, *out_shape), generator=g, dtype=torch.float64) + target_bias

    def __len__(self):
        return self.state.shape[0]

    def __getitem__(self, idx):
        return self.state[idx], self.evaptrans[idx], self.params[idx], self.target[idx]


def make_model_def():
    pressure_names = [f'press_diff_{i}' for i in range(3)]
    evaptrans_names = [f'evaptrans_{i}' for i in range(2)]
    param_names = [f'param_{i}' for i in range(4)]
    scalers = {name: (0.0, 1.0) for name in pressure_names + evaptrans_names + param_names}  # no-op scaling
    return {
        'in_channels': len(pressure_names) + len(evaptrans_names) + len(param_names),
        'out_channels': len(pressure_names),
        'hidden_dim': 8,
        'kernel_size': 3,
        'depth': 1,
        'scalers': scalers,
        'pressure_names': pressure_names,
        'evaptrans_names': evaptrans_names,
        'param_names': param_names,
        'n_evaptrans': len(evaptrans_names),
        'parameter_list': None,
        'param_nlayer': None,
    }


def run_full():
    torch.manual_seed(0)
    model_def = make_model_def()
    out_channels = model_def['out_channels']

    train_ds = SyntheticDataset(200, (3, 16, 16), (2, 16, 16), (4, 16, 16), (out_channels, 16, 16), seed=1)
    valid_ds = SyntheticDataset(40, (3, 16, 16), (2, 16, 16), (4, 16, 16), (out_channels, 16, 16), seed=2)

    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=custom_collate)
    valid_dl = DataLoader(valid_ds, batch_size=16, shuffle=False, collate_fn=custom_collate)
    train_dl_full = DataLoader(train_ds, batch_size=16, shuffle=False, collate_fn=custom_collate)

    net_mean, net_up, net_down = build_networks(model_def)
    net_mean = net_mean.to(torch.float64)
    net_up = net_up.to(torch.float64)
    net_down = net_down.to(torch.float64)

    configs = {
        'quantile': 0.90,
        'max_epochs': {'mean': 15, 'up': 15, 'down': 15},
        'lr': {'mean': 0.01, 'up': 0.01, 'down': 0.01},
        'optimizers': {'mean': 'adam', 'up': 'adam', 'down': 'adam'},
        'early_stop': True,
        'early_stop_start_epoch': 3,
        'wait_patience': 5,
        'restore_best_weights': True,
        'verbose': 1,
    }

    trainer = PI3NNConvTrainer(configs, net_mean, net_up, net_down, train_dl, valid_dl, train_dl_full, device='cpu')
    trainer.train()

    with torch.no_grad():
        for batch in train_dl_full:
            state, evaptrans, params, y = batch
            up_out = trainer.net_up(state, evaptrans, params)
            down_out = trainer.net_down(state, evaptrans, params)
            assert (up_out > 0).all(), 'net_up produced a non-positive output -- PositiveResNetWrapper is broken'
            assert (down_out > 0).all(), 'net_down produced a non-positive output -- PositiveResNetWrapper is broken'
    print('PASS: net_up/net_down outputs are strictly positive.')

    trainer.boundary_optimization(verbose=1)
    results = trainer.evaluate(verbose=1)

    picp_train = results['train']['picp']
    target = configs['quantile']
    max_dev = (picp_train - target).abs().max().item()
    print(f'picp_train per channel: {picp_train.tolist()}, target: {target}, max deviation: {max_dev:.4f}')
    assert max_dev < 0.10, (
        f'PICP on the training set should land close to the target quantile '
        f'(this is what BoundaryOptimizer is defined to do) -- got {picp_train.tolist()}, target {target}'
    )
    print('PASS: per-channel PICP on the training set matches the target quantile.')
    print('PASS: full pipeline sanity checks.')


def get_distributed_info():
    """Inlined copy of main.py's get_distributed_info -- see
    custom_collate's docstring above for why this isn't imported from
    main.py directly. Reads SLURM's env vars (what run_pi3nn_validate.slurm
    launches with via srun, matching every other distributed launch in
    this project) or torchrun's (RANK/WORLD_SIZE/LOCAL_RANK, for ad hoc
    local testing without SLURM)."""
    if 'WORLD_SIZE' in os.environ:
        return int(os.environ['RANK']), int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_RANK'])
    elif 'SLURM_NTASKS' in os.environ:
        return int(os.environ['SLURM_PROCID']), int(os.environ['SLURM_NTASKS']), int(os.environ['SLURM_LOCALID'])
    return 0, 1, 0


def run_ddp_loss_check():
    rank, world_size, _ = get_distributed_info()
    # gloo's env:// rendezvous still needs MASTER_ADDR/MASTER_PORT set
    # (by the launcher, same as run_pi3nn_validate.slurm's srun does via
    # scontrol) even though rank/world_size are passed explicitly here.
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
    assert world_size == 2, 'run_ddp_loss_check expects exactly 2 processes (srun -n2, or torchrun --nproc_per_node=2 for local testing)'

    torch.manual_seed(0)
    full_x = torch.randn(40, dtype=torch.float64)
    full_target = torch.randn(40, dtype=torch.float64)
    # Deliberately uneven mask counts between the two ranks' halves: rank
    # 0 gets 18/20 True, rank 1 gets 3/20 True. If reduced_masked_mse_loss's
    # world_size-cancellation were wrong (e.g. a naive local-mean then
    # relying on DDP's own per-rank averaging), this imbalance is exactly
    # what would produce a detectably biased gradient.
    full_mask = torch.zeros(40, dtype=torch.bool)
    full_mask[torch.randperm(20)[:18]] = True
    full_mask[20 + torch.randperm(20)[:3]] = True

    class TinyLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float64))
            self.b = torch.nn.Parameter(torch.tensor(0.1, dtype=torch.float64))

        def forward(self, x):
            return self.w * x + self.b

    model = TinyLinear()
    ddp_model = DDP(model)  # broadcasts rank 0's initial w/b to rank 1

    local_slice = slice(0, 20) if rank == 0 else slice(20, 40)
    pred_local = ddp_model(full_x[local_slice])
    loss = reduced_masked_mse_loss(pred_local, full_target[local_slice], full_mask[local_slice])
    loss.backward()
    w_grad, b_grad = model.w.grad.item(), model.b.grad.item()

    # Reference: single process, the complete unsharded data, ordinary
    # (non-distributed) masked MSE -- same formula reduced_masked_mse_loss
    # falls back to when dist isn't initialized. Uses the SAME pre-update
    # weights (no optimizer.step() happened above, so model.w/model.b are
    # still the broadcast initial values).
    ref_model = TinyLinear()
    with torch.no_grad():
        ref_model.w.copy_(model.w)
        ref_model.b.copy_(model.b)
    ref_pred = ref_model(full_x)
    ref_loss = ((ref_pred - full_target) ** 2 * full_mask.to(ref_pred.dtype)).sum() / full_mask.sum().clamp_min(1).to(ref_pred.dtype)
    ref_loss.backward()
    ref_w_grad, ref_b_grad = ref_model.w.grad.item(), ref_model.b.grad.item()

    w_diff = abs(w_grad - ref_w_grad)
    b_diff = abs(b_grad - ref_b_grad)
    print(f'[rank {rank}] w_grad={w_grad:.8f} ref={ref_w_grad:.8f} diff={w_diff:.2e} | '
          f'b_grad={b_grad:.8f} ref={ref_b_grad:.8f} diff={b_diff:.2e}')
    assert w_diff < 1e-8 and b_diff < 1e-8, (
        f'rank {rank}: DDP-averaged gradient from reduced_masked_mse_loss does not match '
        f'the single-process reference full-batch gradient (w diff {w_diff:.2e}, b diff {b_diff:.2e})'
    )

    dist.barrier()
    if rank == 0:
        print('PASS: reduced_masked_mse_loss\'s DDP-averaged gradient matches the non-distributed reference.')
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['full', 'ddp_loss'], default='full')
    args = parser.parse_args()
    if args.mode == 'full':
        run_full()
    else:
        run_ddp_loss_check()
