"""
Post-training calibration + evaluation, adapted from
emulator-1ts/pi3nn/trainer.py's boundary_optimization()/evaluate(). Run
once per up_down_mode, after all three trainer.fit() calls (mean, up,
down) for that mode have completed -- not Lightning-managed itself,
just a plain single-process script (this is the "calibrate" phase of
run_pi3nn_phase.py's chained-SLURM-jobs design, so there's no DDP/rank
concept here at all, unlike emulator-1ts's rank-0-only pattern).

Per-channel (not pooled) bisection calibration via the vendored,
unmodified BoundaryOptimizer -- same requirement as emulator-1ts:
pooling all channels together would calibrate to the aggregate
coverage rather than each channel's own, defeating the point of
per-channel calibration. Here "per-channel" flattens across
(batch, timestep, H, W) jointly for each channel, since the extra
timestep axis is just another dimension to flatten away, not a reason
to calibrate separately per-timestep.
"""
import json

import torch
from scipy.fft import dctn, idctn
from torch.utils.data import DataLoader

from .. import model_builder
from ..pfb_dataset import ParFlowSequenceDataset
from ..scalers import create_scalers_from_yaml
from .boundary_optimizer import BoundaryOptimizer
from .networks import build_updown_net, predict_updown


def _load_mean(mean_model_type, mean_model_config, mean_ckpt_path):
    model = model_builder.ModelBuilder.build_emulator(mean_model_type, dict(mean_model_config))
    ckpt = torch.load(mean_ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _load_updown(up_down_mode, updown_config, ckpt_path):
    net = build_updown_net(up_down_mode, updown_config)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    # PI3NNUpDownModule's own checkpoint contains BOTH net_mean.* (its own
    # frozen copy, reloaded separately above anyway) and net.* (the actual
    # trained up/down network) -- extract just the latter and strip the
    # prefix before loading into the bare net built here.
    state = {k[len('net.'):]: v for k, v in ckpt['state_dict'].items() if k.startswith('net.')}
    net.load_state_dict(state)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def _caps(y, mean, up, down, c_up, c_down):
    """y/mean/up/down: 1D numpy arrays, one channel's worth, flattened
    across every sample/timestep/pixel. c_up/c_down: scalars for this
    channel."""
    import numpy as np
    upper = mean + c_up * up
    lower = mean - c_down * down
    inside = (y <= upper) & (y >= lower)
    picp = float(np.mean(inside))
    mpiw = float(np.mean(upper - lower))
    mse = float(np.mean((mean - y) ** 2))
    rmse = mse ** 0.5
    ss_res = np.sum((y - mean) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = float(1 - ss_res / max(ss_tot, 1e-12))
    return {'picp': picp, 'mpiw': mpiw, 'rmse': rmse, 'r2': r2}


def calibrate(config, mean_ckpt_path, up_ckpt_path, down_ckpt_path, up_down_mode, train_dataset=None):
    """train_dataset: bypasses the ParFlowSequenceDataset(data_dir=...)
    construction below when supplied -- same seam train_model() has,
    needed for validate_synthetic.py to run this on in-memory tensors
    instead of real .pfb files."""
    net_mean = _load_mean('ForcedSTRNN', config['mean_model_config'], mean_ckpt_path)
    net_up = _load_updown(up_down_mode, config['updown_model_config'][up_down_mode], up_ckpt_path)
    net_down = _load_updown(up_down_mode, config['updown_model_config'][up_down_mode], down_ckpt_path)

    if train_dataset is not None:
        train_ds = train_dataset
    else:
        scalers = create_scalers_from_yaml(config['scaler_file']) if config.get('scaler_file') else None
        train_ds = ParFlowSequenceDataset(
            data_dir=config['data_dir'], run_name=config['run_name'],
            parameter_list=config['parameter_list'], patch_size=config['patch_size'],
            overlap=config['overlap'], param_nlayer=config['param_nlayer'],
            sequence_length=config['sequence_length'], n_evaptrans=config['n_evaptrans'],
            scalers=scalers, valid_fraction=config['valid_fraction'], split='train',
        )
    train_dl = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    means, ups, downs, ys = [], [], [], []
    with torch.no_grad():
        for batch in train_dl:
            forcing, state, params, target = batch
            pred_up, mean_pred = predict_updown(net_mean, net_up, up_down_mode, forcing, state, params)
            pred_down, _ = predict_updown(net_mean, net_down, up_down_mode, forcing, state, params)
            means.append(mean_pred)
            ups.append(pred_up)
            downs.append(pred_down)
            ys.append(target)
    mean_t = torch.cat(means)
    up_t = torch.cat(ups)
    down_t = torch.cat(downs)
    y_t = torch.cat(ys)

    out_channel = mean_t.shape[2]
    quantile = config['quantile']
    results = {}
    for c in range(out_channel):
        y_c = y_t[:, :, c].flatten().numpy()
        m_c = mean_t[:, :, c].flatten().numpy()
        u_c = up_t[:, :, c].flatten().numpy()
        d_c = down_t[:, :, c].flatten().numpy()
        num_outlier = int(y_c.shape[0] * (1 - quantile) / 2)
        opt = BoundaryOptimizer(
            y_c, m_c, u_c, d_c, num_outlier=num_outlier,
            c_up0_ini=0.0, c_up1_ini=100000.0,
            c_down0_ini=0.0, c_down1_ini=100000.0, max_iter=1000,
        )
        c_up = opt.optimize_up(verbose=0)
        c_down = opt.optimize_down(verbose=0)
        results[f'channel_{c}'] = {'c_up': c_up, 'c_down': c_down, **_caps(y_c, m_c, u_c, d_c, c_up, c_down)}

    # experiment_name, not run_name -- see train.py's data_run_name docstring
    out_path = f"{config['logging_location']}/{config['experiment_name']}_{up_down_mode}_calibration.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved calibration results to {out_path}')
    for k, v in results.items():
        print(k, v)
    return results


# ---------------------------------------------------------------------------
# Spatial-field calibration: c_up/c_down as a smoothed per-cell field instead
# of one scalar per channel. Additive alternative to calibrate() above (left
# completely unchanged) -- selected via config['calibration_mode'] ==
# 'spatial_field' in run_pi3nn_phase.py, not a replacement for the scalar
# baseline. Three stages, each reusing something already validated:
#   1. _fit_raw_cell_field: the existing, unmodified BoundaryOptimizer,
#      called once per spatial cell instead of once globally -- gives a
#      noisy piecewise-constant field (the actual calibration evidence).
#   2. _dct_smooth: closed-form low-pass projection onto a truncated 2D
#      DCT-II basis, via scipy.fft.dctn/idctn. No iteration; rank=1 keeps
#      only the DC coefficient, i.e. the field collapses to its own
#      spatial mean, approximately reproducing calibrate()'s scalar
#      baseline.
#   3. One more global BoundaryOptimizer call, pooled over every point
#      exactly like calibrate() does, but fed up/down pre-multiplied by
#      this point's own smoothed field value -- the returned scalar IS the
#      correction factor that restores the target quantile's coverage
#      globally after stage 2's smoothing perturbed it, with no new
#      numerical method needed.
# ---------------------------------------------------------------------------


def _dct_smooth(field, rank):
    """Project `field` (H, W), possibly containing NaN cells (unvisited
    locations -- only possible in 'absolute' coordinate mode, where
    overlapping-patch coverage is uneven, worse near domain edges), onto
    the first `rank` x `rank` 2D DCT-II low-frequency coefficients
    (scipy.fft.dctn/idctn, norm='ortho' so rank=1 -- keeping only the DC
    coefficient -- collapses the field to its own flat spatial mean,
    same as a plain average). NaN cells are filled with the valid-cell
    mean before projecting -- a simple imputation, not a coverage-
    weighted fit; fine for the moderate gaps boxtest-scale overlap
    produces, worth revisiting (e.g. a weighted least-squares fit) if
    'absolute' mode is ever run on a domain large enough to have
    genuinely sparse corners."""
    import numpy as np
    valid = ~np.isnan(field)
    fill_value = np.nanmean(field) if valid.any() else 0.0
    filled = np.where(valid, field, fill_value)
    H, W = field.shape
    rank_h, rank_w = min(rank, H), min(rank, W)
    coeffs = dctn(filled, norm='ortho')
    mask = np.zeros_like(coeffs, dtype=bool)
    mask[:rank_h, :rank_w] = True
    coeffs = np.where(mask, coeffs, 0.0)
    return idctn(coeffs, norm='ortho')


def _fit_raw_cell_field_loop(y, mean, up, down, cell_index, n_cells, quantile):
    """Reference implementation, kept ONLY for
    validate_synthetic.py's check_spatial_field_vectorized_matches_loop_
    reference to compare against -- production code calls
    _fit_raw_cell_field (the vectorized version below) instead. This is
    what _fit_raw_cell_field originally was: construct the existing,
    unmodified BoundaryOptimizer once per populated cell and loop over
    cells in Python. Correct, but cost scales with n_cells x max_iter x
    (Python call + object-construction overhead) -- fine for
    'patch_relative' mode (n_cells = patch_size**2, small and fixed) but
    the actual scaling bottleneck flagged when spatial-field calibration
    was first built, relevant for 'absolute' mode on anything bigger
    than boxtest scale. See _fit_raw_cell_field's docstring for the fix."""
    import numpy as np
    c_up_raw = np.full(n_cells, np.nan)
    c_down_raw = np.full(n_cells, np.nan)
    order = np.argsort(cell_index, kind='stable')
    sorted_idx = cell_index[order]
    boundaries = np.searchsorted(sorted_idx, np.arange(n_cells + 1))
    for cell in range(n_cells):
        lo, hi = boundaries[cell], boundaries[cell + 1]
        if hi <= lo:
            continue
        pts = order[lo:hi]
        y_c, m_c, u_c, d_c = y[pts], mean[pts], up[pts], down[pts]
        num_outlier = int(y_c.shape[0] * (1 - quantile) / 2)
        opt = BoundaryOptimizer(
            y_c, m_c, u_c, d_c, num_outlier=num_outlier,
            c_up0_ini=0.0, c_up1_ini=100000.0,
            c_down0_ini=0.0, c_down1_ini=100000.0, max_iter=1000,
        )
        c_up_raw[cell] = opt.optimize_up(verbose=0)
        c_down_raw[cell] = opt.optimize_down(verbose=0)
    return c_up_raw, c_down_raw


def _bincount_sum(cell_index, weights, n_cells):
    import numpy as np
    return np.bincount(cell_index, weights=weights.astype(np.float64), minlength=n_cells)


def _vectorized_bisect(y, mean, output, cell_index, n_cells, num_outlier_per_cell, populated, side, max_iter=1000):
    """One bisection search per cell, run for every cell SIMULTANEOUSLY
    via grouped numpy reductions (np.bincount) instead of n_cells
    separate Python-level BoundaryOptimizer instances. Same algorithm,
    same convergence criterion, same search bounds as
    BoundaryOptimizer.optimize_up/optimize_down -- a cell's own (c0, c1,
    f0, f1) state just lives at its own index in a length-n_cells array
    instead of in its own object, and a cell stops being updated (frozen
    at its last c2) once ITS OWN f2 hits exactly 0 or the shared
    iteration budget runs out, via the `active` mask.

    Cost per iteration is one bincount over every point (O(total points),
    not O(n_cells)) -- the fix for the scaling concern _fit_raw_cell_
    field_loop's docstring describes: this version's cost is essentially
    independent of how many distinct cells there are, so 'absolute'
    coordinate mode on a domain far larger than boxtest scale no longer
    pays a per-cell Python-loop penalty. side: 'up' (y >= mean + c*output)
    or 'down' (y <= mean - c*output), matching optimize_up/optimize_down's
    respective directions exactly."""
    import numpy as np
    assert side in ('up', 'down')

    def count(c):
        per_point_c = c[cell_index]
        if side == 'up':
            cond = y >= (mean + per_point_c * output)
        else:
            cond = y <= (mean - per_point_c * output)
        return _bincount_sum(cell_index, cond, n_cells)

    c0 = np.zeros(n_cells)
    c1 = np.full(n_cells, 100000.0)
    f0 = count(c0) - num_outlier_per_cell
    f1 = count(c1) - num_outlier_per_cell
    c2 = c1.copy()
    active = populated & (f0 != 0) & (f1 != 0)
    for _ in range(max_iter):
        if not active.any():
            break
        c2_trial = (c0 + c1) / 2.0
        f2 = count(c2_trial) - num_outlier_per_cell
        c2 = np.where(active, c2_trial, c2)
        move_hi = active & (f2 > 0)   # mirrors BoundaryOptimizer: f2 > 0 moves the c0/c_down0 side up
        hit = active & (f2 == 0)
        move_lo = active & ~move_hi & ~hit
        c0 = np.where(move_hi, c2_trial, c0)
        f0 = np.where(move_hi, f2, f0)
        c1 = np.where(move_lo, c2_trial, c1)
        f1 = np.where(move_lo, f2, f1)
        active = active & ~hit
    return np.where(populated, c2, np.nan)


def _fit_raw_cell_field(y, mean, up, down, cell_index, n_cells, quantile):
    """y/mean/up/down/cell_index: 1D numpy arrays, same length -- one
    entry per (sample, timestep, local-pixel) point. cell_index: integer
    in [0, n_cells) saying which spatial cell this point belongs to.
    Returns (c_up_raw, c_down_raw), each shape (n_cells,), NaN where no
    point landed in that cell -- same contract as the original
    _fit_raw_cell_field_loop (still kept, see its docstring), just
    computed via _vectorized_bisect so cost no longer scales with the
    number of distinct cells. Validated to agree with the loop version
    in validate_synthetic.py's check_spatial_field_vectorized_matches_
    loop_reference."""
    import numpy as np
    counts = _bincount_sum(cell_index, np.ones_like(y, dtype=np.float64), n_cells)
    populated = counts > 0
    num_outlier_per_cell = np.floor(counts * (1 - quantile) / 2)
    c_up_raw = _vectorized_bisect(y, mean, up, cell_index, n_cells, num_outlier_per_cell, populated, side='up')
    c_down_raw = _vectorized_bisect(y, mean, down, cell_index, n_cells, num_outlier_per_cell, populated, side='down')
    return c_up_raw, c_down_raw


def _caps_field(y, mean, up, down, c_up_per_point, c_down_per_point):
    """Same formulas as _caps, but c_up/c_down are already-looked-up
    per-point arrays instead of scalars."""
    import numpy as np
    upper = mean + c_up_per_point * up
    lower = mean - c_down_per_point * down
    inside = (y <= upper) & (y >= lower)
    picp = float(np.mean(inside))
    mpiw = float(np.mean(upper - lower))
    mse = float(np.mean((mean - y) ** 2))
    rmse = mse ** 0.5
    ss_res = np.sum((y - mean) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = float(1 - ss_res / max(ss_tot, 1e-12))
    return picp, mpiw, rmse, r2, inside


def _per_cell_picp_stats(inside, cell_index, n_cells):
    """Diagnostic for spatial-field mode specifically: spread of PICP
    across individual cells, not just the pooled global number -- the
    cheap way to confirm the spatial field is actually more locally
    well-calibrated than the scalar baseline's one-size-fits-all number
    (a low picp_spatial_std means every cell is close to the target
    quantile; the scalar baseline can hit the global target while
    hiding wide per-cell variance)."""
    import numpy as np
    sums = np.bincount(cell_index, weights=inside.astype(float), minlength=n_cells)
    counts = np.bincount(cell_index, minlength=n_cells)
    valid = counts > 0
    per_cell_picp = np.full(n_cells, np.nan)
    per_cell_picp[valid] = sums[valid] / counts[valid]
    return {
        'picp_spatial_mean': float(np.nanmean(per_cell_picp)),
        'picp_spatial_std': float(np.nanstd(per_cell_picp)),
        'picp_spatial_min': float(np.nanmin(per_cell_picp)),
        'picp_spatial_max': float(np.nanmax(per_cell_picp)),
    }


def calibrate_spatial_field(config, mean_ckpt_path, up_ckpt_path, down_ckpt_path, up_down_mode, train_dataset=None):
    """Spatial-field calibration: c_up/c_down become a smoothed per-cell
    field rather than one scalar per channel. See the module-level
    comment above for the three-stage algorithm.

    config['spatial_field_coords']: 'patch_relative' (default) -- field
    indexed by position within each (patch_size, patch_size) sample.
    Needs no dataset changes and has perfectly even cell coverage, since
    every patch contributes exactly one point per relative position by
    construction. 'absolute' -- field indexed by this sample's position
    in the real, fixed CONUS1 domain grid (ParFlowSequenceDataset's
    xbatcher BatchGenerator tiles the same fixed extent every call) --
    needs ParFlowSequenceDataset(return_coords=True); coverage is uneven
    (interior cells get contributions from more overlapping patches than
    edge cells), handled by _dct_smooth's NaN-fill, not a full
    coverage-weighted fit (see that docstring).

    config['spatial_field_rank']: K, the low-pass cutoff per axis
    (default 3, clamped to each axis' own size).

    train_dataset: same dataset-injection seam calibrate() has, for
    validate_synthetic.py to run this on in-memory tensors.
    """
    import numpy as np

    coords_mode = config.get('spatial_field_coords', 'patch_relative')
    if coords_mode not in ('patch_relative', 'absolute'):
        raise ValueError(f"unknown spatial_field_coords {coords_mode!r}")
    rank = config.get('spatial_field_rank', 3)
    quantile = config['quantile']

    net_mean = _load_mean('ForcedSTRNN', config['mean_model_config'], mean_ckpt_path)
    net_up = _load_updown(up_down_mode, config['updown_model_config'][up_down_mode], up_ckpt_path)
    net_down = _load_updown(up_down_mode, config['updown_model_config'][up_down_mode], down_ckpt_path)

    if train_dataset is not None:
        train_ds = train_dataset
    else:
        scalers = create_scalers_from_yaml(config['scaler_file']) if config.get('scaler_file') else None
        train_ds = ParFlowSequenceDataset(
            data_dir=config['data_dir'], run_name=config['run_name'],
            parameter_list=config['parameter_list'], patch_size=config['patch_size'],
            overlap=config['overlap'], param_nlayer=config['param_nlayer'],
            sequence_length=config['sequence_length'], n_evaptrans=config['n_evaptrans'],
            scalers=scalers, valid_fraction=config['valid_fraction'], split='train',
            return_coords=(coords_mode == 'absolute'),
        )
    train_dl = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    means, ups, downs, ys, coords = [], [], [], [], []
    with torch.no_grad():
        for batch in train_dl:
            if coords_mode == 'absolute':
                forcing, state, params, target, coord = batch
                coords.append(coord)
            else:
                forcing, state, params, target = batch
            pred_up, mean_pred = predict_updown(net_mean, net_up, up_down_mode, forcing, state, params)
            pred_down, _ = predict_updown(net_mean, net_down, up_down_mode, forcing, state, params)
            means.append(mean_pred)
            ups.append(pred_up)
            downs.append(pred_down)
            ys.append(target)
    mean_t = torch.cat(means)   # (N, T, C, H, W)
    up_t = torch.cat(ups)
    down_t = torch.cat(downs)
    y_t = torch.cat(ys)
    n_sample, n_time, out_channel, patch_h, patch_w = mean_t.shape

    local_h = np.arange(patch_h)[:, None]
    local_w = np.arange(patch_w)[None, :]
    if coords_mode == 'absolute':
        coord_t = torch.cat(coords).numpy()  # (N, 2) -- (y_min, x_min) per sample
        field_h, field_w = train_ds.Y_EXTENT, train_ds.X_EXTENT
        global_y = coord_t[:, 0][:, None, None] + local_h[None, :, :]
        global_x = coord_t[:, 1][:, None, None] + local_w[None, :, :]
        global_y = np.broadcast_to(global_y, (n_sample, patch_h, patch_w))
        global_x = np.broadcast_to(global_x, (n_sample, patch_h, patch_w))
        cell_hw = (global_y * field_w + global_x).astype(np.int64)
    else:
        field_h, field_w = patch_h, patch_w
        cell_hw = np.broadcast_to((local_h * patch_w + local_w)[None, :, :], (n_sample, patch_h, patch_w)).astype(np.int64)
    n_cells = field_h * field_w
    # Every timestep at a given sample/pixel shares the same spatial cell
    # (T varies the value there, not the location) -- broadcast across T.
    cell_index = np.broadcast_to(cell_hw[:, None, :, :], (n_sample, n_time, patch_h, patch_w)).reshape(-1)

    results = {}
    for c in range(out_channel):
        y_c = y_t[:, :, c].numpy().reshape(-1)
        m_c = mean_t[:, :, c].numpy().reshape(-1)
        u_c = up_t[:, :, c].numpy().reshape(-1)
        d_c = down_t[:, :, c].numpy().reshape(-1)

        c_up_raw, c_down_raw = _fit_raw_cell_field(y_c, m_c, u_c, d_c, cell_index, n_cells, quantile)
        c_up_smooth_field = _dct_smooth(c_up_raw.reshape(field_h, field_w), rank)
        c_down_smooth_field = _dct_smooth(c_down_raw.reshape(field_h, field_w), rank)

        c_up_smooth_per_point = c_up_smooth_field.reshape(-1)[cell_index]
        c_down_smooth_per_point = c_down_smooth_field.reshape(-1)[cell_index]
        num_outlier = int(y_c.shape[0] * (1 - quantile) / 2)
        opt = BoundaryOptimizer(
            y_c, m_c, u_c * c_up_smooth_per_point, d_c * c_down_smooth_per_point,
            num_outlier=num_outlier,
            c_up0_ini=0.0, c_up1_ini=100000.0,
            c_down0_ini=0.0, c_down1_ini=100000.0, max_iter=1000,
        )
        alpha_up = opt.optimize_up(verbose=0)
        alpha_down = opt.optimize_down(verbose=0)
        c_up_field = alpha_up * c_up_smooth_field
        c_down_field = alpha_down * c_down_smooth_field

        c_up_per_point = c_up_field.reshape(-1)[cell_index]
        c_down_per_point = c_down_field.reshape(-1)[cell_index]
        picp, mpiw, rmse, r2, inside = _caps_field(y_c, m_c, u_c, d_c, c_up_per_point, c_down_per_point)
        spatial_stats = _per_cell_picp_stats(inside, cell_index, n_cells)

        results[f'channel_{c}'] = {
            'alpha_up': alpha_up, 'alpha_down': alpha_down,
            'c_up_field': c_up_field.tolist(), 'c_down_field': c_down_field.tolist(),
            'picp': picp, 'mpiw': mpiw, 'rmse': rmse, 'r2': r2,
            **spatial_stats,
        }

    out_path = (
        f"{config['logging_location']}/{config['experiment_name']}_{up_down_mode}_"
        f"{coords_mode}_spatial_calibration.json"
    )
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved spatial-field calibration results to {out_path}')
    for k, v in results.items():
        summary = {kk: vv for kk, vv in v.items() if kk not in ('c_up_field', 'c_down_field')}
        print(k, summary)
    return results
