"""
Small-scale validation for the emulator_configurable PI3NN integration,
mirroring emulator-1ts/pi3nn/validate_synthetic.py's role: a standalone,
runnable correctness check (this repo's test/ package is an empty
placeholder, no pytest infrastructure exists anywhere -- matching
existing convention, not introducing one here) before trusting any of
this on real Frontier-scale CONUS1 data.

Everything emulator-1ts's validate_synthetic.py already covers (the
positive-output requirement, the DDP masked-loss gradient match via a
real DDP-wrapped model + deliberately uneven mask counts, PICP-matches-
quantile) transfers directly here since losses.py/boundary_optimizer.py
are vendored unchanged -- this file focuses on what's genuinely new:
the three-way architecture switch, the return_hidden additive change to
ForcedSTRNN, StatelessUpDownNet's vectorization, freeze correctness
under Lightning's automatic_optimization, recurrent_clone's per-
timestep positivity, and the full mean->up->down->calibrate sequence on
synthetic data via train_model()'s dataset-injection seam.

Usage: python -m emulator_configurable.pi3nn.validate_synthetic
       torchrun --nproc_per_node=2 --standalone -m emulator_configurable.pi3nn.validate_synthetic --mode ddp_loss
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset

from ..models import ForcedSTRNN
from ..train import train_model
from .boundary_optimizer import BoundaryOptimizer
from .calibration import (
    _caps_field, _dct_smooth, _fit_raw_cell_field, _fit_raw_cell_field_loop,
    calibrate, calibrate_spatial_field,
)
from .lightning_modules import PI3NNUpDownModule
from .losses import reduced_masked_mse_loss
from .networks import build_updown_net, predict_updown

UP_DOWN_MODES = ['stateless_output', 'stateless_hidden', 'recurrent_clone']


class SyntheticSequenceDataset(Dataset):
    """Shapes match ParFlowSequenceDataset.__getitem__ exactly. img_channel
    == init_cond_channel == out_channel is required for ForcedSTRNN's
    residual update (x = conv_last(h) + x) to be shape-consistent across
    timesteps -- confirmed from the real boxtest config, where all three
    are set to the same value (5)."""

    def __init__(self, n, sequence_length, act_channel, static_channel, state_channel, patch, seed):
        g = torch.Generator().manual_seed(seed)
        self.forcings = torch.randn(n, sequence_length, act_channel, patch, patch, generator=g)
        self.init_cond = torch.randn(n, 1, state_channel, patch, patch, generator=g)
        self.static_inputs = torch.randn(n, 1, static_channel, patch, patch, generator=g)
        self.target = torch.randn(n, sequence_length, state_channel, patch, patch, generator=g)

    def __len__(self):
        return self.forcings.shape[0]

    def __getitem__(self, idx):
        return self.forcings[idx], self.init_cond[idx], self.static_inputs[idx], self.target[idx]


class SyntheticCoordSequenceDataset(Dataset):
    """Wraps a plain SyntheticSequenceDataset and adds a (y_min, x_min)
    coords tensor plus Y_EXTENT/X_EXTENT attributes -- the same shape
    ParFlowSequenceDataset(return_coords=True) provides, so
    calibrate_spatial_field's 'absolute' coordinate mode can be exercised
    without real .pfb files. Use `.inner` (a plain 4-tuple dataset) for
    training mean/up/down -- their training_step/_step unpack exactly 4
    elements -- and the wrapper itself (5-tuple) only for the calibration
    call, matching how real code only requests coords for calibration."""

    def __init__(self, n, sequence_length, act_channel, static_channel, state_channel, patch, y_extent, x_extent, seed):
        self.inner = SyntheticSequenceDataset(n, sequence_length, act_channel, static_channel, state_channel, patch, seed)
        self.Y_EXTENT = y_extent
        self.X_EXTENT = x_extent
        g = torch.Generator().manual_seed(seed + 1000)
        y_min = torch.randint(0, y_extent - patch + 1, (n,), generator=g)
        x_min = torch.randint(0, x_extent - patch + 1, (n,), generator=g)
        self.coords = torch.stack([y_min, x_min], dim=1)

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        forcings, init_cond, static_inputs, target = self.inner[idx]
        return forcings, init_cond, static_inputs, target, self.coords[idx]


def make_mean_config():
    return dict(num_layers=1, num_hidden=[8], img_channel=3, act_channel=2, init_cond_channel=3, static_channel=4, out_channel=3)


def make_updown_config(up_down_mode, mean_config):
    if up_down_mode == 'stateless_output':
        return dict(act_channel=2, static_channel=4, out_channel=3, hidden_dim=8, kernel_size=3, depth=1)
    elif up_down_mode == 'stateless_hidden':
        return dict(
            act_channel=2, static_channel=4, out_channel=3,
            hidden_state_channel=mean_config['num_hidden'][-1],
            hidden_dim=8, kernel_size=3, depth=1,
        )
    elif up_down_mode == 'recurrent_clone':
        return dict(
            num_layers=1, num_hidden=[8], act_channel=2,
            init_cond_channel=3, static_channel=4, out_channel=3,
        )
    raise ValueError(up_down_mode)


def check_positivity_all_modes():
    torch.manual_seed(0)
    mean_config = make_mean_config()
    net_mean = ForcedSTRNN(**mean_config).eval()
    for p in net_mean.parameters():
        p.requires_grad_(False)

    forcing = torch.randn(2, 4, mean_config['act_channel'], 8, 8)
    state = torch.randn(2, 1, mean_config['init_cond_channel'], 8, 8)
    params = torch.randn(2, 1, mean_config['static_channel'], 8, 8)

    for mode in UP_DOWN_MODES:
        net = build_updown_net(mode, make_updown_config(mode, mean_config))
        pred, _ = predict_updown(net_mean, net, mode, forcing, state, params)
        assert (pred > 0).all(), f'{mode}: output not strictly positive'
    print('PASS: strictly-positive output for all three up_down_mode variants.')


def check_return_hidden_preserves_behavior():
    torch.manual_seed(1)
    mean_config = make_mean_config()
    net_mean = ForcedSTRNN(**mean_config).eval()

    forcing = torch.randn(2, 4, mean_config['act_channel'], 8, 8)
    state = torch.randn(2, 1, mean_config['init_cond_channel'], 8, 8)
    params = torch.randn(2, 1, mean_config['static_channel'], 8, 8)

    with torch.no_grad():
        out_plain = net_mean(forcing, state, params)
        out_pair = net_mean(forcing, state, params, return_hidden=True)

    assert isinstance(out_pair, tuple) and len(out_pair) == 2
    assert torch.equal(out_plain, out_pair[0]), 'return_hidden=False call site is not behavior-preserving'
    expected_hidden_shape = (2, 4, mean_config['num_hidden'][-1], 8, 8)
    assert out_pair[1].shape == expected_hidden_shape, f'hidden shape {out_pair[1].shape} != {expected_hidden_shape}'
    print('PASS: return_hidden=False is bit-identical to the original call site; hidden tensor shape correct.')


def check_stateless_vectorization():
    torch.manual_seed(2)
    mean_config = make_mean_config()
    net = build_updown_net('stateless_output', make_updown_config('stateless_output', mean_config)).eval()

    batch, T, H, W = 2, 4, 8, 8
    forcings = torch.randn(batch, T, mean_config['act_channel'], H, W)
    static_inputs = torch.randn(batch, 1, mean_config['static_channel'], H, W)
    mean_out = torch.randn(batch, T, mean_config['out_channel'], H, W)

    with torch.no_grad():
        vectorized = net(forcings, static_inputs, mean_out)

        static = static_inputs[:, 0]
        ref_frames = []
        for t in range(T):
            x = torch.cat([forcings[:, t], static, mean_out[:, t]], dim=1)
            ref_frames.append(net.positive(net.backbone(x)))
        reference = torch.stack(ref_frames, dim=1)

    assert torch.allclose(vectorized, reference, atol=1e-6), 'vectorized StatelessUpDownNet path disagrees with per-timestep reference loop'
    print('PASS: StatelessUpDownNet vectorization matches a per-timestep reference loop.')


def check_freeze_correctness():
    torch.manual_seed(3)
    mean_config = make_mean_config()
    net_mean = ForcedSTRNN(**mean_config)
    with tempfile.TemporaryDirectory() as d:
        ckpt_path = os.path.join(d, 'mean.ckpt')
        torch.save({'state_dict': net_mean.state_dict()}, ckpt_path)

        updown_mode = 'stateless_output'
        module = PI3NNUpDownModule(
            up_down_mode=updown_mode, role='up',
            mean_model_type='ForcedSTRNN', mean_model_config=mean_config, mean_ckpt_path=ckpt_path,
            updown_config=make_updown_config(updown_mode, mean_config),
        )
        module.learning_rate = 0.01
        module.train()

        before = {k: v.clone() for k, v in module.net_mean.state_dict().items()}

        batch = (
            torch.randn(2, 4, mean_config['act_channel'], 8, 8),
            torch.randn(2, 1, mean_config['init_cond_channel'], 8, 8),
            torch.randn(2, 1, mean_config['static_channel'], 8, 8),
            torch.randn(2, 4, mean_config['out_channel'], 8, 8),
        )
        optimizer = module.configure_optimizers()
        for _ in range(3):
            optimizer.zero_grad()
            loss = module.training_step(batch, 0)
            loss.backward()
            optimizer.step()

        after = module.net_mean.state_dict()
        for k in before:
            assert torch.equal(before[k], after[k]), f'net_mean.{k} changed after training_step -- freeze is broken'
        assert module.net_mean.training is False, 'net_mean left in train mode after module.train()'
    print('PASS: net_mean is bit-identical and stays in eval mode after training_steps (freeze correctness).')


def check_recurrent_clone_positivity():
    torch.manual_seed(4)
    mean_config = make_mean_config()
    net = build_updown_net('recurrent_clone', make_updown_config('recurrent_clone', mean_config))

    forcing = torch.randn(2, 5, mean_config['act_channel'], 8, 8)
    state = torch.randn(2, 1, mean_config['init_cond_channel'], 8, 8)
    params = torch.randn(2, 1, mean_config['static_channel'], 8, 8)
    with torch.no_grad():
        out = net(forcing, state, params)
    # out already contains every timestep's fed-forward x stacked (dim=1)
    # -- positivity of the whole tensor IS positivity at every timestep.
    assert (out > 0).all(), 'recurrent_clone is not strictly positive at every timestep'
    print('PASS: recurrent_clone output is strictly positive at every timestep.')


def check_full_sequence():
    torch.manual_seed(5)
    mean_config = make_mean_config()
    seq_len = 3
    train_ds = SyntheticSequenceDataset(16, seq_len, mean_config['act_channel'], mean_config['static_channel'], mean_config['out_channel'], 8, seed=10)
    valid_ds = SyntheticSequenceDataset(8, seq_len, mean_config['act_channel'], mean_config['static_channel'], mean_config['out_channel'], 8, seed=11)

    common = dict(
        data_dir='', parameter_list=[], param_nlayer=[], patch_size=0, overlap=0,
        sequence_length=seq_len, batch_size=4, num_workers=0, precision='32',
        device='cpu', gradient_loss_penalty=False, logging_frequency=1,
        valid_fraction=0.0, early_stopping_patience=None,
    )

    with tempfile.TemporaryDirectory() as logdir:
        mean_ckpt = train_model(
            run_name='synthetic_mean', model_type='ForcedSTRNN', model_config=mean_config,
            max_epochs=2, learning_rate=0.01, logging_location=logdir,
            train_dataset=train_ds, valid_dataset=valid_ds, **common,
        )
        assert mean_ckpt, 'mean phase produced no checkpoint path -- likely every_n_train_steps too high for this tiny dataset'

        for up_down_mode in UP_DOWN_MODES:
            updown_config = make_updown_config(up_down_mode, mean_config)
            role_ckpts = {}
            for role in ('up', 'down'):
                role_ckpts[role] = train_model(
                    run_name=f'synthetic_{up_down_mode}_{role}', model_type='PI3NNUpDownModule',
                    model_config={
                        'up_down_mode': up_down_mode, 'role': role,
                        'mean_model_type': 'ForcedSTRNN', 'mean_model_config': mean_config,
                        'mean_ckpt_path': mean_ckpt, 'updown_config': updown_config,
                    },
                    max_epochs=2, learning_rate=0.01, logging_location=logdir,
                    train_dataset=train_ds, valid_dataset=valid_ds, **common,
                )
                assert role_ckpts[role], f'{up_down_mode}/{role} produced no checkpoint path'

            results = calibrate(
                config={
                    'mean_model_config': mean_config,
                    'updown_model_config': {up_down_mode: updown_config},
                    'quantile': 0.9, 'logging_location': logdir, 'experiment_name': f'synthetic_{up_down_mode}',
                    'batch_size': common['batch_size'], 'num_workers': common['num_workers'],
                },
                mean_ckpt_path=mean_ckpt, up_ckpt_path=role_ckpts['up'], down_ckpt_path=role_ckpts['down'],
                up_down_mode=up_down_mode, train_dataset=train_ds,
            )
            assert len(results) == mean_config['out_channel']
            print(f'[{up_down_mode}] calibration results: {results}')

    print('PASS: full mean -> up -> down -> calibrate sequence completes for all three up_down_mode variants.')


def check_run_pi3nn_phase_registers_updown():
    """Regression test for an actual bug hit on Frontier: model_builder's
    @register_emulator('PI3NNUpDownModule') decorator only runs when
    lightning_modules.py is imported, and run_pi3nn_phase.py's own import
    chain didn't do that -- model_setup() raised
    KeyError('PI3NNUpDownModule') the moment the up/down phase actually
    tried to build one. Every other check in this file is insensitive to
    this, because this file ALSO (independently) imports
    lightning_modules.PI3NNUpDownModule at module level for
    check_freeze_correctness -- by the time any check function here runs,
    the registration already happened for an unrelated reason, masking
    exactly this gap. Needs a genuinely fresh subprocess that imports
    ONLY run_pi3nn_phase to actually test it."""
    result = subprocess.run(
        [sys.executable, '-c',
         "import emulator_configurable.pi3nn.run_pi3nn_phase as m\n"
         "from emulator_configurable.model_builder import ModelBuilder\n"
         "assert 'PI3NNUpDownModule' in ModelBuilder.registry['emulator']"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f'importing run_pi3nn_phase alone did not register PI3NNUpDownModule:\n{result.stderr}'
    )
    print('PASS: importing run_pi3nn_phase alone registers PI3NNUpDownModule (no hidden import-order dependency).')


def check_dct_smooth_rank1_is_spatial_mean():
    """rank=1 keeps only the DC coefficient -- algebraically this must
    collapse the field to its own flat spatial mean everywhere (the
    rank=1 case is what makes spatial-field calibration a strict
    generalization of calibrate()'s scalar baseline, not a parallel
    mechanism)."""
    rng = np.random.RandomState(0)
    field = rng.randn(6, 7)
    smoothed = _dct_smooth(field, 1)
    assert np.allclose(smoothed, field.mean(), atol=1e-8), 'rank=1 DCT smoothing should collapse to the spatial mean'
    print('PASS: spatial-field rank=1 collapses to the field\'s own spatial mean (scalar-baseline equivalent).')


def check_spatial_field_vectorized_matches_loop_reference():
    """_fit_raw_cell_field was rewritten to run every cell's bisection
    search simultaneously via np.bincount (cost ~independent of n_cells)
    instead of one BoundaryOptimizer object per cell in a Python loop
    (cost scales with n_cells) -- the fix for the scalability concern
    flagged when 'absolute' coordinate mode was designed. Confirms the
    faster version agrees with the original, still-correct loop
    implementation (kept as _fit_raw_cell_field_loop) on the same random
    multi-cell data, uneven cell sizes included (not every cell gets the
    same point count, matching 'absolute' mode's real uneven coverage)."""
    rng = np.random.RandomState(123)
    n_cells = 40
    cell_index, y, mean, up, down = [], [], [], [], []
    for cell in range(n_cells):
        # Deliberately uneven per-cell counts (20-120 points) -- the
        # thing that makes 'absolute' mode harder than 'patch_relative'.
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
    """Isolates stages 1-2 (per-cell raw fit + DCT smoothing) from the
    rest of the pipeline using hand-built synthetic data with a KNOWN
    smooth per-cell noise scale, net_up/net_down held at a constant 1 so
    the raw per-cell c_up IS (up to sampling noise and a shared constant)
    proportional to the true scale. Checks that a higher-rank smoothed
    field actually resembles that known shape, not just that the code
    runs."""
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


def check_spatial_field_patch_relative_integration():
    """End-to-end integration through the real dataloader/network stack
    (not hand-built arrays like the checks above) -- reuses
    check_full_sequence's pattern: train mean -> up -> down, then run
    calibrate_spatial_field in 'patch_relative' mode (the default, no
    coords needed) and sanity-check the result shape."""
    torch.manual_seed(50)
    mean_config = make_mean_config()
    seq_len = 3
    patch = 8
    train_ds = SyntheticSequenceDataset(16, seq_len, mean_config['act_channel'], mean_config['static_channel'], mean_config['out_channel'], patch, seed=60)

    common = dict(
        data_dir='', parameter_list=[], param_nlayer=[], patch_size=0, overlap=0,
        sequence_length=seq_len, batch_size=4, num_workers=0, precision='32',
        device='cpu', gradient_loss_penalty=False, logging_frequency=1,
        valid_fraction=0.0, early_stopping_patience=None,
    )
    with tempfile.TemporaryDirectory() as logdir:
        mean_ckpt = train_model(
            run_name='synthetic_spatial_mean', model_type='ForcedSTRNN', model_config=mean_config,
            max_epochs=1, learning_rate=0.01, logging_location=logdir,
            train_dataset=train_ds, valid_dataset=train_ds, **common,
        )
        up_down_mode = 'stateless_output'
        updown_config = make_updown_config(up_down_mode, mean_config)
        role_ckpts = {}
        for role in ('up', 'down'):
            role_ckpts[role] = train_model(
                run_name=f'synthetic_spatial_{role}', model_type='PI3NNUpDownModule',
                model_config={
                    'up_down_mode': up_down_mode, 'role': role,
                    'mean_model_type': 'ForcedSTRNN', 'mean_model_config': mean_config,
                    'mean_ckpt_path': mean_ckpt, 'updown_config': updown_config,
                },
                max_epochs=1, learning_rate=0.01, logging_location=logdir,
                train_dataset=train_ds, valid_dataset=train_ds, **common,
            )
        results = calibrate_spatial_field(
            config={
                'mean_model_config': mean_config,
                'updown_model_config': {up_down_mode: updown_config},
                'quantile': 0.9, 'logging_location': logdir, 'experiment_name': 'synthetic_spatial',
                'batch_size': common['batch_size'], 'num_workers': common['num_workers'],
                'spatial_field_rank': 2,
            },
            mean_ckpt_path=mean_ckpt, up_ckpt_path=role_ckpts['up'], down_ckpt_path=role_ckpts['down'],
            up_down_mode=up_down_mode, train_dataset=train_ds,
        )
        assert len(results) == mean_config['out_channel']
        for ch, res in results.items():
            field = np.array(res['c_up_field'])
            assert field.shape == (patch, patch), f'{ch}: c_up_field shape {field.shape} != ({patch},{patch})'
            assert 'picp_field' in res, f'{ch}: calibrate_spatial_field results missing picp_field'
            assert np.array(res['picp_field']).shape == (patch, patch)
        print("PASS: calibrate_spatial_field runs end-to-end in 'patch_relative' mode (default) through the real pipeline.")

        from .plot_spatial_field import load_and_plot_json
        json_path = f'{logdir}/synthetic_spatial_{up_down_mode}_patch_relative_spatial_calibration.json'
        png_path = f'{logdir}/check.png'
        load_and_plot_json(json_path, out_path=png_path, quantile=0.9)
        assert os.path.exists(png_path) and os.path.getsize(png_path) > 0, 'plot_spatial_field produced no (or an empty) PNG'
    print('PASS: plot_spatial_field.load_and_plot_json renders a real calibrate_spatial_field() JSON result without error.')


def check_spatial_field_absolute_mode():
    """Same integration shape as check_spatial_field_patch_relative_
    integration, but exercises 'absolute' coordinate mode via
    SyntheticCoordSequenceDataset -- confirms ParFlowSequenceDataset's
    new return_coords plumbing and the global-domain field assembly in
    calibrate_spatial_field actually wire together, producing a field
    sized to the full synthetic domain rather than one patch."""
    torch.manual_seed(51)
    mean_config = make_mean_config()
    seq_len = 2
    patch = 4
    y_extent, x_extent = 10, 10
    train_ds = SyntheticCoordSequenceDataset(
        40, seq_len, mean_config['act_channel'], mean_config['static_channel'], mean_config['out_channel'],
        patch, y_extent, x_extent, seed=70,
    )

    common = dict(
        data_dir='', parameter_list=[], param_nlayer=[], patch_size=0, overlap=0,
        sequence_length=seq_len, batch_size=4, num_workers=0, precision='32',
        device='cpu', gradient_loss_penalty=False, logging_frequency=1,
        valid_fraction=0.0, early_stopping_patience=None,
    )
    with tempfile.TemporaryDirectory() as logdir:
        mean_ckpt = train_model(
            run_name='synthetic_abs_mean', model_type='ForcedSTRNN', model_config=mean_config,
            max_epochs=1, learning_rate=0.01, logging_location=logdir,
            train_dataset=train_ds.inner, valid_dataset=train_ds.inner, **common,
        )
        up_down_mode = 'stateless_output'
        updown_config = make_updown_config(up_down_mode, mean_config)
        role_ckpts = {}
        for role in ('up', 'down'):
            role_ckpts[role] = train_model(
                run_name=f'synthetic_abs_{role}', model_type='PI3NNUpDownModule',
                model_config={
                    'up_down_mode': up_down_mode, 'role': role,
                    'mean_model_type': 'ForcedSTRNN', 'mean_model_config': mean_config,
                    'mean_ckpt_path': mean_ckpt, 'updown_config': updown_config,
                },
                max_epochs=1, learning_rate=0.01, logging_location=logdir,
                train_dataset=train_ds.inner, valid_dataset=train_ds.inner, **common,
            )
        results = calibrate_spatial_field(
            config={
                'mean_model_config': mean_config,
                'updown_model_config': {up_down_mode: updown_config},
                'quantile': 0.9, 'logging_location': logdir, 'experiment_name': 'synthetic_abs',
                'batch_size': common['batch_size'], 'num_workers': common['num_workers'],
                'spatial_field_coords': 'absolute', 'spatial_field_rank': 2,
            },
            mean_ckpt_path=mean_ckpt, up_ckpt_path=role_ckpts['up'], down_ckpt_path=role_ckpts['down'],
            up_down_mode=up_down_mode, train_dataset=train_ds,
        )
        assert len(results) == mean_config['out_channel']
        for ch, res in results.items():
            field = np.array(res['c_up_field'])
            assert field.shape == (y_extent, x_extent), f'{ch}: c_up_field shape {field.shape} != ({y_extent},{x_extent})'
    print("PASS: calibrate_spatial_field runs end-to-end in 'absolute' coordinate mode, producing a full-domain field.")


def run_full():
    check_positivity_all_modes()
    check_return_hidden_preserves_behavior()
    check_stateless_vectorization()
    check_freeze_correctness()
    check_recurrent_clone_positivity()
    check_full_sequence()
    check_run_pi3nn_phase_registers_updown()
    check_dct_smooth_rank1_is_spatial_mean()
    check_spatial_field_vectorized_matches_loop_reference()
    check_spatial_field_recovers_smooth_pattern()
    check_spatial_field_global_coverage_restored()
    check_spatial_field_patch_relative_integration()
    check_spatial_field_absolute_mode()
    print('PASS: all synthetic validation checks passed.')


def run_ddp_loss_check():
    """Identical in spirit to emulator-1ts/pi3nn/validate_synthetic.py's
    ddp_loss check -- losses.py is vendored unchanged, so this is the
    same check, just kept here too so this package's validation is
    self-contained."""
    if 'WORLD_SIZE' in os.environ:
        rank, world_size = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    elif 'SLURM_NTASKS' in os.environ:
        rank, world_size = int(os.environ['SLURM_PROCID']), int(os.environ['SLURM_NTASKS'])
    else:
        rank, world_size = 0, 1
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
    assert world_size == 2, 'run_ddp_loss_check expects exactly 2 processes'

    torch.manual_seed(0)
    full_x = torch.randn(40, dtype=torch.float64)
    full_target = torch.randn(40, dtype=torch.float64)
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
    ddp_model = DDP(model)

    local_slice = slice(0, 20) if rank == 0 else slice(20, 40)
    pred_local = ddp_model(full_x[local_slice])
    loss = reduced_masked_mse_loss(pred_local, full_target[local_slice], full_mask[local_slice])
    loss.backward()
    w_grad, b_grad = model.w.grad.item(), model.b.grad.item()

    ref_model = TinyLinear()
    with torch.no_grad():
        ref_model.w.copy_(model.w)
        ref_model.b.copy_(model.b)
    ref_pred = ref_model(full_x)
    ref_loss = ((ref_pred - full_target) ** 2 * full_mask.to(ref_pred.dtype)).sum() / full_mask.sum().clamp_min(1).to(ref_pred.dtype)
    ref_loss.backward()
    ref_w_grad, ref_b_grad = ref_model.w.grad.item(), ref_model.b.grad.item()

    w_diff, b_diff = abs(w_grad - ref_w_grad), abs(b_grad - ref_b_grad)
    print(f'[rank {rank}] w_grad diff={w_diff:.2e} b_grad diff={b_diff:.2e}')
    assert w_diff < 1e-8 and b_diff < 1e-8
    dist.barrier()
    if rank == 0:
        print("PASS: reduced_masked_mse_loss's DDP-averaged gradient matches the non-distributed reference.")
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['full', 'ddp_loss'], default='full')
    args = parser.parse_args()
    if args.mode == 'full':
        run_full()
    else:
        run_ddp_loss_check()
