"""
Assemble a full-domain prediction+bounds+MSE plot by running net_mean/
net_up/net_down over every patch tiling the WHOLE real domain at one
timestep, averaging predictions in overlapping regions -- instead of
plotting a single patch_size x patch_size sample the way
plot_prediction.py does. Only meaningful for 'absolute'-mode
checkpoints: c_up_field/c_down_field need to be indexed by real domain
position for "the whole domain" to mean anything; patch_relative mode's
field is indexed by position-within-patch instead.

The network itself (ResNet: Conv2d + per-pixel-channel LayerNorm2D, no
Linear/pooling layers) is fully shape-agnostic and could in principle
take the whole domain in a single forward pass -- deliberately NOT done
here. It was only ever trained on patch_size crops, so a single-pass
whole-domain forward would be an untested distribution shift (every
interior pixel would see more real context than it ever did in
training); stitching individually patch-sized forward passes together
instead exactly reproduces what training/calibration actually saw.

--overlap here is independent of whatever overlap the checkpoint's own
training data used -- it only controls how much blending THIS stitching
pass does between adjacent patches' predictions. Start at --overlap 0
(each real pixel covered by exactly one patch, averaging is a no-op) to
see raw per-patch discontinuities at the seams, then increase it to see
how much blending smooths them out.

Usage:
  python3 -m pi3nn.plot_stitched_prediction --pth /path/to/NAME_pi3nn.pth \
      --config /path/to/pi3nn_fullyear_config.yaml \
      --split train --index 0 --overlap 0 [--out /path/to/output.png]
"""
import argparse

import matplotlib
matplotlib.use('Agg')  # headless (Frontier batch jobs have no display) -- must precede pyplot import
import numpy as np
import torch
import yaml

from pi3nn.networks import build_networks
from pi3nn.plot_prediction import compute_bounds, plot_prediction_fields
from utils import get_dtype

# ParFlowDataset is imported lazily inside load_and_plot (below), not here --
# same reasoning as plot_prediction.py's lazy import.


def patch_starts(extent, patch_size, overlap):
    """Grid of patch start positions tiling [0, extent) with patch_size
    x patch_size patches and the given overlap -- always includes a
    final position clamped to extent - patch_size, guaranteeing full
    coverage even when (extent - patch_size) isn't evenly divisible by
    the stride. This is the exact xbatcher gap this project already
    found and worked around for 'absolute'-mode calibration coverage --
    this script reuses the fix rather than reintroducing the bug."""
    stride = patch_size - overlap
    assert stride > 0, f'overlap ({overlap}) must be smaller than patch_size ({patch_size})'
    starts = list(range(0, extent - patch_size, stride))
    last = extent - patch_size
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


def seam_discontinuity_stats(field, patch_size, overlap):
    """Quantifies whether patch_starts' internal seams (the boundary
    between adjacent patches) show a measurably larger jump in `field`
    than typical interior (non-seam) adjacent pixels do -- the direct,
    numeric version of eyeballing a stitched plot for grid-aligned
    discontinuities, since a real but modest jump can easily hide under
    a plot's colorscale without visually dominating.

    field: (C, H, W). Most informative at overlap=0, where a seam is a
    genuine abrupt juxtaposition of two independently-predicted patches;
    at higher overlap the seam band is smoothed by averaging, so the
    real comparison is against overlap=0's own numbers for the SAME
    checkpoint (does the ratio shrink as overlap grows?), not just
    looking at one overlap value in isolation.

    Returns {'y': (seam_jump, interior_jump), 'x': (seam_jump,
    interior_jump)}, each shape (C,) -- seam_jump is the mean absolute
    difference across every internal seam position (NOT the true domain
    edges -- those aren't a stitching artifact), interior_jump is the
    same statistic over every other adjacent-pixel pair, excluding seam
    positions entirely so it isn't contaminated by the very thing being
    measured against it."""
    C, H, W = field.shape
    results = {}
    for axis, extent in (('y', H), ('x', W)):
        starts = patch_starts(extent, patch_size, overlap)
        seam_positions = sorted(s for s in set(starts) if 0 < s < extent)
        diffs = np.abs(np.diff(field, axis=1 if axis == 'y' else 2))  # diffs[..., i] = |field[..., i+1] - field[..., i]|
        n_diffs = diffs.shape[1 if axis == 'y' else 2]
        is_seam = np.zeros(n_diffs, dtype=bool)
        for s in seam_positions:
            is_seam[s - 1] = True  # diffs index s-1 straddles the seam AT s (field[s] vs field[s-1])
        reduce_axes = (1, 2)
        if axis == 'y':
            seam_jump = diffs[:, is_seam, :].mean(axis=reduce_axes)
            interior_jump = diffs[:, ~is_seam, :].mean(axis=reduce_axes)
        else:
            seam_jump = diffs[:, :, is_seam].mean(axis=reduce_axes)
            interior_jump = diffs[:, :, ~is_seam].mean(axis=reduce_axes)
        results[axis] = (seam_jump, interior_jump)
    return results


def load_and_plot(pth_path, config_path, split='train', index=0, overlap=0, out_path=None, dataset=None, physical_units=True):
    """dataset: same dataset-injection seam the other pi3nn plotting
    scripts have, for synthetic testing."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    save_dict = torch.load(pth_path, map_location='cpu')
    model_def = save_dict['model_def']
    dtype = get_dtype(config.get('dtype', 'float32'))

    if save_dict.get('spatial_field_coords') != 'absolute':
        raise ValueError(
            f"{pth_path} is not an 'absolute'-mode checkpoint "
            f"(spatial_field_coords={save_dict.get('spatial_field_coords')!r}) -- a stitched "
            f"full-domain plot only makes sense when c_up_field/c_down_field are indexed by "
            f"real domain position."
        )

    bias_init = save_dict.get('bias_init', 3.0)
    eps = save_dict.get('eps', 0.2)
    print(f'[plot_stitched_prediction] loaded bias_init={bias_init!r}, eps={eps!r} from {pth_path}')
    net_mean, net_up, net_down = build_networks(model_def, bias_init=bias_init, eps=eps)
    net_mean.load_state_dict(save_dict['net_mean'])
    net_up.load_state_dict(save_dict['net_up'])
    net_down.load_state_dict(save_dict['net_down'])
    net_mean = net_mean.to(dtype).eval()
    net_up = net_up.to(dtype).eval()
    net_down = net_down.to(dtype).eval()

    if dataset is not None:
        ds = dataset
    else:
        from dataset import ParFlowDataset
        data_def = dict(config['data_def'])
        data_def.pop('scaler_yaml', None)
        valid_fraction = config.get('valid_fraction', 0.1)
        ds = ParFlowDataset(**data_def, dtype=dtype, valid_fraction=valid_fraction, split=split)

    # Resolve --index to a real timestep the same way plot_residual_field.py
    # does -- ds.bgen is shuffled, so index doesn't correspond to a
    # sequential/predictable point in time on its own.
    time_index = ds.bgen[index]['time'].values[0]
    print(f'[plot_stitched_prediction] --index {index} (split={split!r}) resolved to '
          f'time_index={time_index}, file={ds.pressure_files["t"][time_index]}')

    patch_size = ds.patch_size
    out_channels = model_def['out_channels']
    Y_EXTENT, X_EXTENT = ds.Y_EXTENT, ds.X_EXTENT

    c_up_full = save_dict['c_up_field'].numpy()
    c_down_full = save_dict['c_down_field'].numpy()

    y_starts = patch_starts(Y_EXTENT, patch_size, overlap)
    x_starts = patch_starts(X_EXTENT, patch_size, overlap)
    print(f'[plot_stitched_prediction] domain {Y_EXTENT}x{X_EXTENT}, patch_size={patch_size}, '
          f'overlap={overlap}: {len(y_starts)}x{len(x_starts)} = {len(y_starts) * len(x_starts)} patches')

    sum_mean = np.zeros((out_channels, Y_EXTENT, X_EXTENT))
    sum_upper = np.zeros_like(sum_mean)
    sum_lower = np.zeros_like(sum_mean)
    sum_target = np.zeros_like(sum_mean)
    count = np.zeros((Y_EXTENT, X_EXTENT))

    for y_min in y_starts:
        for x_min in x_starts:
            state, evaptrans, params, target = ds.read_patch(time_index, y_min, x_min)
            # .clone(): see plot_prediction.py's load_and_plot for why --
            # scale_pressure/unscale_* below mutate in place, and would
            # otherwise corrupt a reused injected dataset's storage.
            state, evaptrans, params, target = (t.unsqueeze(0).clone() for t in (state, evaptrans, params, target))
            net_mean.scale_pressure(state)
            net_mean.scale_evaptrans(evaptrans)
            net_mean.scale_statics(params)

            with torch.no_grad():
                mean_pred_t = net_mean(state, evaptrans, params)
                up_pred_t = net_up(state, evaptrans, params)
                down_pred_t = net_down(state, evaptrans, params)

            if physical_units:
                net_mean.unscale_pressure(mean_pred_t)
                net_mean.unscale_pressure_width(up_pred_t)
                net_mean.unscale_pressure_width(down_pred_t)
            else:
                net_mean.scale_pressure(target)

            mean_pred = mean_pred_t[0].numpy()
            up_pred = up_pred_t[0].numpy()
            down_pred = down_pred_t[0].numpy()
            target_np = target[0].numpy()

            c_up = c_up_full[:, y_min:y_min + patch_size, x_min:x_min + patch_size]
            c_down = c_down_full[:, y_min:y_min + patch_size, x_min:x_min + patch_size]
            upper, lower, _ = compute_bounds(mean_pred, up_pred, down_pred, c_up, c_down)

            sl = (slice(None), slice(y_min, y_min + patch_size), slice(x_min, x_min + patch_size))
            sum_mean[sl] += mean_pred
            sum_upper[sl] += upper
            sum_lower[sl] += lower
            sum_target[sl] += target_np
            count[y_min:y_min + patch_size, x_min:x_min + patch_size] += 1

    assert (count > 0).all(), 'some domain cells were never covered by any patch -- patch_starts has a gap'
    print(f'[plot_stitched_prediction] per-pixel patch coverage: min={count.min():.0f}, '
          f'max={count.max():.0f}, mean={count.mean():.2f}')

    mean_pred = sum_mean / count
    upper = sum_upper / count
    lower = sum_lower / count
    target_np = sum_target / count
    width = upper - lower
    squared_error = (target_np - mean_pred) ** 2

    for c in range(out_channels):
        print(f'[plot_stitched_prediction] channel {c}: MSE (whole domain) = {squared_error[c].mean():.4g}')

    # Quantifies whether patch seams show a measurably larger jump in the
    # mean prediction than typical interior pixels do -- the numeric
    # version of eyeballing the plot for grid-aligned discontinuities
    # (a real but modest jump can hide under a colorscale without
    # visually dominating). Most informative at overlap=0; the real
    # comparison is against this same printout at a higher --overlap for
    # the SAME checkpoint, to see whether blending actually shrinks it.
    seam_stats = seam_discontinuity_stats(mean_pred, patch_size, overlap)
    for axis in ('y', 'x'):
        seam_jump, interior_jump = seam_stats[axis]
        for c in range(out_channels):
            ratio = seam_jump[c] / max(interior_jump[c], 1e-12)
            print(f'[plot_stitched_prediction] channel {c}, {axis}-seams: mean |jump| at seams={seam_jump[c]:.4g}, '
                  f'at interior pixels={interior_jump[c]:.4g} (ratio {ratio:.2f}x)')

    channel_names = model_def.get('pressure_names')
    out_path = out_path or pth_path.rsplit('.pth', 1)[0] + f'_{split}_idx{index}_overlap{overlap}_stitched.png'
    plot_prediction_fields(mean_pred, upper, lower, width, squared_error, out_path, channel_names=channel_names)
    return mean_pred, upper, lower, width, squared_error


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth', required=True, help="Path to an 'absolute'-mode {name}_pi3nn.pth file")
    parser.add_argument('--config', required=True, help='The same yaml config used for training (for data_def)')
    parser.add_argument('--split', choices=['train', 'valid'], default='train')
    parser.add_argument('--index', type=int, default=0, help='Which sample to resolve a real timestep from')
    parser.add_argument('--overlap', type=int, default=0,
                         help="Patch overlap for THIS stitching pass -- independent of the "
                              "checkpoint's own training overlap. Start at 0 to see raw seams.")
    parser.add_argument('--out', default=None, help='Output PNG path (default: alongside --pth)')
    parser.add_argument('--scaled', action='store_true',
                         help='Plot in standardized (z-score) units instead of physical pressure units')
    args = parser.parse_args()
    load_and_plot(args.pth, args.config, split=args.split, index=args.index, overlap=args.overlap,
                  out_path=args.out, physical_units=not args.scaled)
