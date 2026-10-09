"""
Spatial-field calibration helpers: c_up/c_down as a smoothed per-cell
field instead of one scalar per channel. Vendored from
emulator_configurable/pi3nn/calibration.py's corresponding section --
these functions are pure numpy/scipy (take already-flattened y/mean/up/
down/cell_index arrays), not tied to either package's model/dataset
classes, so the port is a direct copy rather than a redesign. Kept as a
separate file (not folded into trainer.py) so the two integrations'
copies stay easy to diff against each other if a bug is ever found in
one -- fix it in both places, this is a point-in-time copy, not a live
link (same vendoring rationale as boundary_optimizer.py).

Three-stage design, each stage reusing something already validated:
  1. _fit_raw_cell_field: the existing, unmodified BoundaryOptimizer,
     run per spatial cell (vectorized across all cells via np.bincount,
     not a Python loop -- see _fit_raw_cell_field_loop's docstring for
     the reference version and the honest, scale-dependent benchmark
     this was checked against).
  2. _dct_smooth: closed-form low-pass projection onto a truncated 2D
     DCT-II basis via scipy.fft.dctn/idctn. No iteration; rank=1 keeps
     only the DC coefficient, i.e. the field collapses to its own
     spatial mean, approximately reproducing boundary_optimization()'s
     scalar baseline.
  3. One more global BoundaryOptimizer call, pooled over every point
     exactly like boundary_optimization() does, but fed up/down pre-
     multiplied by this point's own smoothed field value -- the
     returned scalar IS the correction factor that restores the target
     quantile's coverage globally after stage 2's smoothing perturbed
     it, with no new numerical method needed.
"""
import numpy as np
from scipy.fft import dctn, idctn

from .boundary_optimizer import BoundaryOptimizer


def _bincount_sum(cell_index, weights, n_cells):
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
    iteration budget runs out, via the `active` mask. side: 'up'
    (y >= mean + c*output) or 'down' (y <= mean - c*output), matching
    optimize_up/optimize_down's respective directions exactly."""
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


def _fit_raw_cell_field_loop(y, mean, up, down, cell_index, n_cells, quantile):
    """Reference implementation, kept ONLY for validate_synthetic.py's
    check_spatial_field_vectorized_matches_loop_reference to compare
    against -- production code calls _fit_raw_cell_field (the vectorized
    version below) instead. Construct the existing, unmodified
    BoundaryOptimizer once per populated cell and loop over cells in
    Python. Correct, but cost scales with n_cells x max_iter x (Python
    call + object-construction overhead)."""
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


def _fit_raw_cell_field(y, mean, up, down, cell_index, n_cells, quantile):
    """y/mean/up/down/cell_index: 1D numpy arrays, same length -- one
    entry per (sample, local-pixel) point. cell_index: integer in
    [0, n_cells) saying which spatial cell this point belongs to.
    Returns (c_up_raw, c_down_raw), each shape (n_cells,), NaN where no
    point landed in that cell. Validated to agree with the loop version
    (_fit_raw_cell_field_loop) in validate_synthetic.py's
    check_spatial_field_vectorized_matches_loop_reference."""
    counts = _bincount_sum(cell_index, np.ones_like(y, dtype=np.float64), n_cells)
    populated = counts > 0
    num_outlier_per_cell = np.floor(counts * (1 - quantile) / 2)
    c_up_raw = _vectorized_bisect(y, mean, up, cell_index, n_cells, num_outlier_per_cell, populated, side='up')
    c_down_raw = _vectorized_bisect(y, mean, down, cell_index, n_cells, num_outlier_per_cell, populated, side='down')
    return c_up_raw, c_down_raw


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


def _caps_field(y, mean, up, down, c_up_per_point, c_down_per_point):
    """Same formulas as the scalar baseline's per-channel PICP/MPIW/
    RMSE/R2, but c_up/c_down are already-looked-up per-point arrays
    instead of scalars."""
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
    well-calibrated than the scalar baseline's one-size-fits-all
    number."""
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


def fit_spatial_field(y, mean, up, down, cell_index, n_cells, field_shape, quantile, rank, verbose=0):
    """Orchestrates the three stages for ONE channel's already-flattened
    data and returns (c_up_field, c_down_field, alpha_up, alpha_down),
    each field shaped `field_shape` (= (patch_size, patch_size) for
    'patch_relative', or (Y_EXTENT, X_EXTENT) for 'absolute'). Shared by
    both integrations' trainer-level orchestration code (trainer.py here,
    calibration.py in emulator_configurable) since everything below this
    point is dataset/model-agnostic."""
    c_up_raw, c_down_raw = _fit_raw_cell_field(y, mean, up, down, cell_index, n_cells, quantile)
    c_up_smooth_field = _dct_smooth(c_up_raw.reshape(field_shape), rank)
    c_down_smooth_field = _dct_smooth(c_down_raw.reshape(field_shape), rank)

    c_up_smooth_pp = c_up_smooth_field.reshape(-1)[cell_index]
    c_down_smooth_pp = c_down_smooth_field.reshape(-1)[cell_index]
    num_outlier = int(y.shape[0] * (1 - quantile) / 2)
    opt = BoundaryOptimizer(
        y, mean, up * c_up_smooth_pp, down * c_down_smooth_pp,
        num_outlier=num_outlier,
        c_up0_ini=0.0, c_up1_ini=100000.0,
        c_down0_ini=0.0, c_down1_ini=100000.0, max_iter=1000,
    )
    alpha_up = opt.optimize_up(verbose=0)
    alpha_down = opt.optimize_down(verbose=0)

    if verbose > 0:
        # alpha hitting EXACTLY the search ceiling (100000.0) means the
        # bisection's own upper endpoint f1 never reached 0 -- i.e. even
        # the widest allowed rescale couldn't bring the outside-count down
        # to num_outlier. Two different reasons produce that symptom and
        # need different fixes: (a) the smoothed field going negative or
        # near-zero SOMEWHERE (breaks the monotonicity BoundaryOptimizer's
        # docstring explicitly requires -- increasing alpha then makes
        # that cell's bound narrower, not wider, so it can actively
        # prevent convergence), or (b) the field staying strictly positive
        # everywhere but some points' true residual is large enough that
        # no finite rescale (up to the 100000 ceiling) covers them AND
        # there are more such points than num_outlier allows (a genuine
        # data/undersampling issue, not a smoothing bug). Printing the
        # smoothed field's own min (sign/near-zero check) plus the
        # actual achieved vs. target outside-count at the ceiling
        # distinguishes them directly instead of guessing.
        up_outside_at_ceiling = int(np.count_nonzero(y >= mean + 100000.0 * (up * c_up_smooth_pp)))
        down_outside_at_ceiling = int(np.count_nonzero(y <= mean - 100000.0 * (down * c_down_smooth_pp)))
        print(f'[fit_spatial_field] smoothed field min/max: c_up [{c_up_smooth_field.min():.4g}, {c_up_smooth_field.max():.4g}], '
              f'c_down [{c_down_smooth_field.min():.4g}, {c_down_smooth_field.max():.4g}]')
        print(f'[fit_spatial_field] alpha_up={alpha_up:.4f}, alpha_down={alpha_down:.4f}, num_outlier target={num_outlier} '
              f'(total points={y.shape[0]}); at the ceiling (alpha=100000): up_outside={up_outside_at_ceiling}, '
              f'down_outside={down_outside_at_ceiling}')

    c_up_field = alpha_up * c_up_smooth_field
    c_down_field = alpha_down * c_down_smooth_field
    return c_up_field, c_down_field, alpha_up, alpha_down
