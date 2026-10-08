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
import tempfile

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

from .networks import build_networks
from .trainer import PI3NNConvTrainer
from .losses import reduced_masked_mse_loss
from .boundary_optimizer import BoundaryOptimizer
from .spatial_calibration import (
    _caps_field, _dct_smooth, _fit_raw_cell_field, _fit_raw_cell_field_loop,
)


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
        # Real ParFlowDataset always sets this; boundary_optimization_spatial_
        # field's 'patch_relative' mode reads it off loader.dataset to size
        # the field before the rank-0-only full pass runs -- this test
        # double needs to honor that same minimal attribute contract.
        self.patch_size = out_shape[-1]

    def __len__(self):
        return self.state.shape[0]

    def __getitem__(self, idx):
        return self.state[idx], self.evaptrans[idx], self.params[idx], self.target[idx]


class SyntheticCoordDataset(Dataset):
    """Wraps a plain SyntheticDataset and adds a (y_min, x_min) coords
    tensor plus Y_EXTENT/X_EXTENT/patch_size attributes -- the same
    shape ParFlowDataset(return_coords=True) provides, so
    boundary_optimization_spatial_field's 'absolute' coordinate mode can
    be exercised without real .pfb files. Use `.inner` (a plain 4-tuple
    dataset) for training mean/up/down; the wrapper itself (5-tuple)
    only for the coords-bearing loaders, matching how real code only
    requests coords for spatial-field calibration specifically."""

    def __init__(self, n, state_shape, evaptrans_shape, params_shape, out_shape, patch, y_extent, x_extent, seed):
        self.inner = SyntheticDataset(n, state_shape, evaptrans_shape, params_shape, out_shape, seed)
        self.Y_EXTENT = y_extent
        self.X_EXTENT = x_extent
        self.patch_size = patch
        g = torch.Generator().manual_seed(seed + 1000)
        y_min = torch.randint(0, y_extent - patch + 1, (n,), generator=g)
        x_min = torch.randint(0, x_extent - patch + 1, (n,), generator=g)
        self.coords = torch.stack([y_min, x_min], dim=1)

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        state, evaptrans, params, target = self.inner[idx]
        return state, evaptrans, params, target, self.coords[idx]


def custom_collate_with_coords(batch):
    s, e, p, y, c = [], [], [], [], []
    for b in batch:
        s.append(b[0])
        e.append(b[1])
        p.append(b[2])
        y.append(b[3])
        c.append(b[4])
    return torch.stack(s), torch.stack(e), torch.stack(p), torch.stack(y), torch.stack(c)


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

    check_compute_bounds_broadcasting()
    check_dct_smooth_rank1_is_spatial_mean()
    check_spatial_field_vectorized_matches_loop_reference()
    check_spatial_field_recovers_smooth_pattern()
    check_spatial_field_global_coverage_restored()
    check_spatial_field_patch_relative_integration()
    check_spatial_field_absolute_mode()
    print('PASS: full pipeline sanity checks.')


def check_compute_bounds_broadcasting():
    """plot_prediction.compute_bounds must handle both calibration_mode
    outputs: a scalar c_up/c_down (one constant per channel, shape
    (C,)) and a spatial field (one value per cell, shape (C, H, W)).
    Checked directly against a hand-computed reference rather than just
    'does it run', since silent incorrect broadcasting (e.g. a scalar
    accidentally broadcasting against the wrong axis) would be easy to
    get wrong and hard to notice visually in a rendered plot."""
    from .plot_prediction import compute_bounds
    rng = np.random.RandomState(0)
    C, H, W = 3, 4, 5
    mean_pred = rng.randn(C, H, W)
    up_pred = np.abs(rng.randn(C, H, W)) + 0.1
    down_pred = np.abs(rng.randn(C, H, W)) + 0.1

    c_up_scalar = np.array([1.0, 2.0, 3.0])
    c_down_scalar = np.array([0.5, 1.5, 2.5])
    upper, lower, width = compute_bounds(mean_pred, up_pred, down_pred, c_up_scalar, c_down_scalar)
    for c in range(C):
        expected_upper = mean_pred[c] + c_up_scalar[c] * up_pred[c]
        expected_lower = mean_pred[c] - c_down_scalar[c] * down_pred[c]
        assert np.allclose(upper[c], expected_upper), f'scalar mode: upper mismatch on channel {c}'
        assert np.allclose(lower[c], expected_lower), f'scalar mode: lower mismatch on channel {c}'
    assert np.allclose(width, upper - lower)
    print('PASS: compute_bounds broadcasts a per-channel scalar c_up/c_down correctly.')

    c_up_field = rng.rand(C, H, W) + 0.5
    c_down_field = rng.rand(C, H, W) + 0.5
    upper, lower, width = compute_bounds(mean_pred, up_pred, down_pred, c_up_field, c_down_field)
    expected_upper = mean_pred + c_up_field * up_pred
    expected_lower = mean_pred - c_down_field * down_pred
    assert np.allclose(upper, expected_upper), 'field mode: upper mismatch'
    assert np.allclose(lower, expected_lower), 'field mode: lower mismatch'
    assert np.allclose(width, upper - lower)
    print('PASS: compute_bounds broadcasts a per-cell field c_up/c_down correctly.')


def check_dct_smooth_rank1_is_spatial_mean():
    """rank=1 keeps only the DC coefficient -- algebraically this must
    collapse the field to its own flat spatial mean everywhere (the
    rank=1 case is what makes spatial-field calibration a strict
    generalization of boundary_optimization()'s scalar baseline, not a
    parallel mechanism)."""
    rng = np.random.RandomState(0)
    field = rng.randn(6, 7)
    smoothed = _dct_smooth(field, 1)
    assert np.allclose(smoothed, field.mean(), atol=1e-8), 'rank=1 DCT smoothing should collapse to the spatial mean'
    print('PASS: spatial-field rank=1 collapses to the field\'s own spatial mean (scalar-baseline equivalent).')


def check_spatial_field_vectorized_matches_loop_reference():
    """_fit_raw_cell_field runs every cell's bisection search
    simultaneously via np.bincount (cost ~independent of n_cells)
    instead of one BoundaryOptimizer object per cell in a Python loop
    (cost scales with n_cells). Confirms the faster version agrees with
    the original, still-correct loop implementation
    (_fit_raw_cell_field_loop) on the same random multi-cell data,
    uneven cell sizes included (not every cell gets the same point
    count, matching 'absolute' mode's real uneven coverage)."""
    rng = np.random.RandomState(123)
    n_cells = 40
    cell_index, y, mean, up, down = [], [], [], [], []
    for cell in range(n_cells):
        n_pts = rng.randint(20, 120)
        scale = 0.5 + rng.rand()
        y.append(rng.randn(n_pts) * scale)
        mean.append(rng.randn(n_pts) * 0.1)
        up.append(np.abs(rng.randn(n_pts)) + 0.5)
        down.append(np.abs(rng.randn(n_pts)) + 0.5)
        cell_index.append(np.full(n_pts, cell))
    y, mean = np.concatenate(y), np.concatenate(mean)
    up, down = np.concatenate(up), np.concatenate(down)
    cell_index = np.concatenate(cell_index).astype(np.int64)

    c_up_vec, c_down_vec = _fit_raw_cell_field(y, mean, up, down, cell_index, n_cells, quantile=0.9)
    c_up_loop, c_down_loop = _fit_raw_cell_field_loop(y, mean, up, down, cell_index, n_cells, quantile=0.9)

    assert np.allclose(c_up_vec, c_up_loop, atol=1e-6), 'vectorized c_up disagrees with the loop reference'
    assert np.allclose(c_down_vec, c_down_loop, atol=1e-6), 'vectorized c_down disagrees with the loop reference'
    print('PASS: vectorized per-cell bisection matches the original per-cell-loop reference exactly.')


def check_spatial_field_recovers_smooth_pattern():
    """Isolates stages 1-2 (per-cell raw fit + DCT smoothing) using
    hand-built synthetic data with a KNOWN smooth per-cell noise scale,
    net_up/net_down held at a constant 1 so the raw per-cell c_up IS (up
    to sampling noise and a shared constant) proportional to the true
    scale. Checks a higher-rank smoothed field actually resembles that
    known shape, not just that the code runs."""
    rng = np.random.RandomState(42)
    H, W = 6, 6
    n_per_cell = 400
    hh = np.arange(H)
    true_scale = 1.0 + 0.8 * np.cos(np.pi * hh / H)  # smooth along H, constant along W
    true_scale_field = np.broadcast_to(true_scale[:, None], (H, W))

    cell_index, y, mean, up, down = [], [], [], [], []
    for h in range(H):
        for w in range(W):
            cell = h * W + w
            noise = rng.randn(n_per_cell) * true_scale[h]
            y.append(noise)
            mean.append(np.zeros(n_per_cell))
            up.append(np.ones(n_per_cell))
            down.append(np.ones(n_per_cell))
            cell_index.append(np.full(n_per_cell, cell))
    y, mean = np.concatenate(y), np.concatenate(mean)
    up, down = np.concatenate(up), np.concatenate(down)
    cell_index = np.concatenate(cell_index).astype(np.int64)

    c_up_raw, _ = _fit_raw_cell_field(y, mean, up, down, cell_index, H * W, quantile=0.9)
    smoothed = _dct_smooth(c_up_raw.reshape(H, W), rank=3)

    true_dev = (true_scale_field - true_scale_field.mean()).flatten()
    smooth_dev = (smoothed - smoothed.mean()).flatten()
    cos_sim = np.dot(true_dev, smooth_dev) / (np.linalg.norm(true_dev) * np.linalg.norm(smooth_dev) + 1e-12)
    assert cos_sim > 0.8, f'smoothed field does not resemble the known smooth pattern (cosine similarity={cos_sim:.3f})'
    print(f'PASS: rank=3 spatial field recovers the known smooth pattern shape (cosine similarity={cos_sim:.3f}).')


def check_spatial_field_global_coverage_restored():
    """Stage 3 check: smoothing (stage 2) perturbs each cell away from
    its own stage-1 optimum, which can drag the pooled global coverage
    off the target quantile. Confirms the one extra global
    BoundaryOptimizer call (reused unmodified, fed up/down pre-
    multiplied by the smoothed field) actually restores it."""
    rng = np.random.RandomState(7)
    H, W = 5, 5
    n_per_cell = 300
    quantile = 0.9
    hh = np.arange(H)
    true_scale = 1.0 + 0.6 * np.sin(np.pi * hh / H)

    cell_index, y, mean, up, down = [], [], [], [], []
    for h in range(H):
        for w in range(W):
            cell = h * W + w
            noise = rng.randn(n_per_cell) * true_scale[h]
            y.append(noise)
            mean.append(np.zeros(n_per_cell))
            up.append(np.ones(n_per_cell))
            down.append(np.ones(n_per_cell))
            cell_index.append(np.full(n_per_cell, cell))
    y, mean = np.concatenate(y), np.concatenate(mean)
    up, down = np.concatenate(up), np.concatenate(down)
    cell_index = np.concatenate(cell_index).astype(np.int64)
    n_cells = H * W

    c_up_raw, c_down_raw = _fit_raw_cell_field(y, mean, up, down, cell_index, n_cells, quantile)
    c_up_smooth = _dct_smooth(c_up_raw.reshape(H, W), rank=2)
    c_down_smooth = _dct_smooth(c_down_raw.reshape(H, W), rank=2)

    c_up_smooth_pp = c_up_smooth.reshape(-1)[cell_index]
    c_down_smooth_pp = c_down_smooth.reshape(-1)[cell_index]
    num_outlier = int(y.shape[0] * (1 - quantile) / 2)
    opt = BoundaryOptimizer(
        y, mean, up * c_up_smooth_pp, down * c_down_smooth_pp, num_outlier=num_outlier,
        c_up0_ini=0.0, c_up1_ini=100000.0, c_down0_ini=0.0, c_down1_ini=100000.0, max_iter=1000,
    )
    alpha_up, alpha_down = opt.optimize_up(), opt.optimize_down()
    c_up_field, c_down_field = alpha_up * c_up_smooth, alpha_down * c_down_smooth

    c_up_final_pp = c_up_field.reshape(-1)[cell_index]
    c_down_final_pp = c_down_field.reshape(-1)[cell_index]
    picp, mpiw, rmse, r2, inside = _caps_field(y, mean, up, down, c_up_final_pp, c_down_final_pp)
    assert abs(picp - quantile) < 0.02, f'global PICP {picp:.4f} not close to target quantile {quantile} after stage-3 rescale'
    print(f'PASS: stage-3 global rescale restores target coverage after smoothing (picp={picp:.4f}, target={quantile}).')


def _assert_spatial_picp_stats_sane(results, out_channels, field_shape):
    """Regression check for a real gap caught by the user: evaluate_
    spatial_field's results must include per-cell PICP spread
    (picp_spatial_mean/std/min/max) AND the per-cell map itself
    (picp_field), not just the pooled global picp -- _per_cell_picp_
    stats-equivalent reporting was ported into spatial_calibration.py but
    never actually wired into trainer.py's evaluate path until this was
    fixed. min <= mean <= max and std >= 0 are real algebraic
    requirements of how these are computed (a bounds check, not just
    presence), and picp_field's own (non-NaN) min/max must be consistent
    with the reported scalar min/max."""
    keys = ('picp_spatial_mean', 'picp_spatial_std', 'picp_spatial_min', 'picp_spatial_max')
    for split in ('train', 'valid'):
        for k in keys:
            assert k in results[split], f"evaluate_spatial_field()'s {split!r} results missing {k!r}"
            v = results[split][k]
            assert v.shape == (out_channels,), f'{split}/{k} shape {v.shape} != ({out_channels},)'
        mean, std, lo, hi = (results[split][k] for k in keys)
        assert (std >= 0).all(), f'{split}: picp_spatial_std has a negative entry'
        assert (lo <= mean).all() and (mean <= hi).all(), f'{split}: picp_spatial_min/mean/max out of order'

        assert 'picp_field' in results[split], f"evaluate_spatial_field()'s {split!r} results missing 'picp_field'"
        field = results[split]['picp_field']
        assert field.shape == (out_channels, *field_shape), f'{split}/picp_field shape {field.shape} != ({out_channels}, {field_shape})'
        for c in range(out_channels):
            vals = field[c][~torch.isnan(field[c])]
            assert vals.numel() > 0, f'{split}/picp_field channel {c} is entirely NaN'
            assert abs(vals.min().item() - lo[c].item()) < 1e-6, f'{split}/picp_field channel {c} min disagrees with picp_spatial_min'
            assert abs(vals.max().item() - hi[c].item()) < 1e-6, f'{split}/picp_field channel {c} max disagrees with picp_spatial_max'


def check_spatial_field_patch_relative_integration():
    """End-to-end integration through the real PI3NNConvTrainer pipeline
    (not hand-built arrays like the checks above): train -> spatial-field
    boundary_optimization -> spatial-field evaluate, in 'patch_relative'
    mode (the default, no coords needed)."""
    torch.manual_seed(50)
    model_def = make_model_def()
    out_channels = model_def['out_channels']
    patch = 8

    train_ds = SyntheticDataset(60, (3, patch, patch), (2, patch, patch), (4, patch, patch), (out_channels, patch, patch), seed=60)
    valid_ds = SyntheticDataset(20, (3, patch, patch), (2, patch, patch), (4, patch, patch), (out_channels, patch, patch), seed=61)
    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=custom_collate)
    valid_dl = DataLoader(valid_ds, batch_size=8, shuffle=False, collate_fn=custom_collate)
    train_dl_full = DataLoader(train_ds, batch_size=8, shuffle=False, collate_fn=custom_collate)

    net_mean, net_up, net_down = build_networks(model_def)
    net_mean, net_up, net_down = (n.to(torch.float64) for n in (net_mean, net_up, net_down))

    configs = {
        'quantile': 0.9,
        'max_epochs': {'mean': 3, 'up': 3, 'down': 3},
        'lr': {'mean': 0.01, 'up': 0.01, 'down': 0.01},
        'optimizers': {'mean': 'adam', 'up': 'adam', 'down': 'adam'},
        'early_stop': False,
        'wait_patience': 5,
        'restore_best_weights': False,
        'verbose': 0,
    }

    trainer = PI3NNConvTrainer(configs, net_mean, net_up, net_down, train_dl, valid_dl, train_dl_full, device='cpu')
    trainer.train()
    trainer.boundary_optimization_spatial_field(coords_mode='patch_relative', rank=2, verbose=0)
    results = trainer.evaluate_spatial_field(verbose=0)

    assert trainer.c_up_field.shape == (out_channels, patch, patch), f'c_up_field shape {trainer.c_up_field.shape} != ({out_channels},{patch},{patch})'
    assert 'train' in results and 'valid' in results
    _assert_spatial_picp_stats_sane(results, out_channels, (patch, patch))
    print("PASS: boundary_optimization_spatial_field/evaluate_spatial_field run end-to-end in 'patch_relative' mode.")

    from .plot_prediction import load_and_plot as load_and_plot_prediction
    with tempfile.TemporaryDirectory() as d:
        pth_path = os.path.join(d, 'synthetic_pi3nn.pth')
        torch.save({
            'net_mean': net_mean.state_dict(), 'net_up': net_up.state_dict(), 'net_down': net_down.state_dict(),
            'model_def': model_def, 'c_up_field': trainer.c_up_field, 'c_down_field': trainer.c_down_field,
        }, pth_path)
        config_path = os.path.join(d, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump({'dtype': 'float64'}, f)
        png_path = os.path.join(d, 'check_prediction.png')
        load_and_plot_prediction(pth_path, config_path, split='train', index=0, out_path=png_path, dataset=train_ds)
        assert os.path.exists(png_path) and os.path.getsize(png_path) > 0, 'plot_prediction produced no (or an empty) PNG'
    print('PASS: plot_prediction.load_and_plot renders mean/bounds for a real sample without error.')

    from .plot_spatial_field import load_and_plot_pth
    with tempfile.TemporaryDirectory() as d:
        pth_path = os.path.join(d, 'synthetic_pi3nn.pth')
        torch.save({
            'c_up_field': trainer.c_up_field, 'c_down_field': trainer.c_down_field,
            'spatial_field_coords': 'patch_relative', 'results': results, 'model_def': model_def,
        }, pth_path)
        png_path = os.path.join(d, 'check.png')
        load_and_plot_pth(pth_path, split='train', out_path=png_path, quantile=0.9)
        assert os.path.exists(png_path) and os.path.getsize(png_path) > 0, 'plot_spatial_field produced no (or an empty) PNG'
    print('PASS: plot_spatial_field.load_and_plot_pth renders a real {name}_pi3nn.pth result without error.')


def check_spatial_field_absolute_mode():
    """Same integration shape as check_spatial_field_patch_relative_
    integration, but exercises 'absolute' coordinate mode via
    SyntheticCoordDataset -- confirms ParFlowDataset's new return_coords
    plumbing and the full-domain field assembly in
    boundary_optimization_spatial_field/evaluate_spatial_field actually
    wire together, producing a field sized to the full synthetic domain
    rather than one patch."""
    torch.manual_seed(51)
    model_def = make_model_def()
    out_channels = model_def['out_channels']
    patch = 4
    y_extent, x_extent = 10, 10

    train_coord_ds = SyntheticCoordDataset(40, (3, patch, patch), (2, patch, patch), (4, patch, patch), (out_channels, patch, patch), patch, y_extent, x_extent, seed=70)
    valid_coord_ds = SyntheticCoordDataset(16, (3, patch, patch), (2, patch, patch), (4, patch, patch), (out_channels, patch, patch), patch, y_extent, x_extent, seed=71)

    train_dl = DataLoader(train_coord_ds.inner, batch_size=8, shuffle=True, collate_fn=custom_collate)
    valid_dl = DataLoader(valid_coord_ds.inner, batch_size=8, shuffle=False, collate_fn=custom_collate)
    train_dl_full = DataLoader(train_coord_ds.inner, batch_size=8, shuffle=False, collate_fn=custom_collate)
    train_dl_full_coords = DataLoader(train_coord_ds, batch_size=8, shuffle=False, collate_fn=custom_collate_with_coords)
    valid_dl_coords = DataLoader(valid_coord_ds, batch_size=8, shuffle=False, collate_fn=custom_collate_with_coords)

    net_mean, net_up, net_down = build_networks(model_def)
    net_mean, net_up, net_down = (n.to(torch.float64) for n in (net_mean, net_up, net_down))

    configs = {
        'quantile': 0.9,
        'max_epochs': {'mean': 3, 'up': 3, 'down': 3},
        'lr': {'mean': 0.01, 'up': 0.01, 'down': 0.01},
        'optimizers': {'mean': 'adam', 'up': 'adam', 'down': 'adam'},
        'early_stop': False,
        'wait_patience': 5,
        'restore_best_weights': False,
        'verbose': 0,
    }

    trainer = PI3NNConvTrainer(
        configs, net_mean, net_up, net_down, train_dl, valid_dl, train_dl_full, device='cpu',
        train_dl_full_coords=train_dl_full_coords, valid_dl_coords=valid_dl_coords,
    )
    trainer.train()
    trainer.boundary_optimization_spatial_field(coords_mode='absolute', rank=2, verbose=0)
    results = trainer.evaluate_spatial_field(verbose=0)

    assert trainer.c_up_field.shape == (out_channels, y_extent, x_extent), f'c_up_field shape {trainer.c_up_field.shape} != ({out_channels},{y_extent},{x_extent})'
    assert 'train' in results and 'valid' in results
    _assert_spatial_picp_stats_sane(results, out_channels, (y_extent, x_extent))
    print("PASS: boundary_optimization_spatial_field/evaluate_spatial_field run end-to-end in 'absolute' coordinate mode, full-domain field.")


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
