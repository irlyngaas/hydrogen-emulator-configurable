"""
Plot one real input sample's mean prediction plus calibrated upper/
lower bounds and interval width, as per-channel heatmap panels --
complements plot_spatial_field.py (which plots the calibration
CONSTANTS: c_up_field/c_down_field/picp_field) by showing what those
constants actually produce for a real forecast.

Separate panels for mean/lower/upper/width rather than per-pixel error
bars: drawing a literal error-bar glyph at every one of patch_size**2
(or Y_EXTENT x X_EXTENT) points would be unreadable -- side-by-side
heatmaps is the standard way 2D spatial uncertainty gets visualized.

Works with EITHER calibration_mode's saved output: scalar c_up/c_down
(one constant per channel) or the spatial field c_up_field/c_down_field
(one value per cell) -- compute_bounds broadcasts either shape
correctly, dispatching on which key is present in the .pth.

Usage:
  python3 -m pi3nn.plot_prediction --pth /path/to/NAME_pi3nn.pth \
      --config /path/to/pi3nn_boxtest_config.yaml \
      --split train --index 0 [--out /path/to/output.png]
"""
import argparse

import matplotlib
matplotlib.use('Agg')  # headless (Frontier batch jobs have no display) -- must precede pyplot import
import numpy as np
import torch
import yaml

from pi3nn.networks import build_networks
from utils import get_dtype

# ParFlowDataset is imported lazily inside load_and_plot (below), not here --
# dataset.py eagerly imports xarray/xbatcher at module level, which would
# otherwise be required just to import THIS module even when a dataset is
# injected (validate_synthetic.py's synthetic check never constructs a real
# ParFlowDataset at all). Same reasoning as pfb_dataset.py's lazy imports in
# emulator_configurable.


def compute_bounds(mean_pred, up_pred, down_pred, c_up, c_down):
    """mean_pred/up_pred/down_pred: (out_channels, H, W) numpy arrays
    for ONE sample. c_up/c_down: either a per-channel scalar (shape
    (out_channels,)) or a per-channel field (shape (out_channels, H, W))
    -- broadcasts correctly either way. Returns (upper, lower, width),
    each (out_channels, H, W)."""
    c_up = np.asarray(c_up)
    c_down = np.asarray(c_down)
    if c_up.ndim == 1:
        c_up = c_up[:, None, None]
        c_down = c_down[:, None, None]
    upper = mean_pred + c_up * up_pred
    lower = mean_pred - c_down * down_pred
    width = upper - lower
    return upper, lower, width


def plot_prediction_fields(mean_pred, upper, lower, width, out_path, channel_names=None):
    """mean_pred/upper/lower/width: (out_channels, H, W) numpy arrays.
    Saves one PNG, one row per channel, four columns: mean (its own
    color scale), down-offset (mean - lower), up-offset (upper - mean),
    width (upper - lower). Plotting raw lower/upper on the SAME scale as
    mean (an earlier version of this function did) breaks badly
    whenever the calibrated bound is narrow relative to the mean
    field's own spatial range -- which is the common case, since the
    bound only needs to span the residual, not the signal -- making
    mean/lower/upper look visually identical and the plot uninformative
    (caught by the user looking at a real rendered plot). Plotting the
    OFFSETS instead, on their own shared scale, shows the actual
    spatial pattern in the bounds regardless of how it compares to
    mean's own range, and additionally reveals asymmetry between the
    up and down bound that width alone collapses away."""
    import matplotlib.pyplot as plt

    out_channels = mean_pred.shape[0]
    channel_names = channel_names or [f'channel {c}' for c in range(out_channels)]
    fig, axes = plt.subplots(out_channels, 4, figsize=(16, 3.2 * out_channels), squeeze=False)

    for c in range(out_channels):
        ax_mean, ax_down_off, ax_up_off, ax_width = axes[c]

        im_mean = ax_mean.imshow(mean_pred[c], cmap='viridis')
        ax_mean.set_title(f'{channel_names[c]}: mean')
        fig.colorbar(im_mean, ax=ax_mean, fraction=0.046, pad=0.04)

        down_offset = mean_pred[c] - lower[c]   # >= 0
        up_offset = upper[c] - mean_pred[c]     # >= 0
        # A single outlier pixel (plausibly a patch-edge artifact) can
        # dominate a raw max() and wash out everywhere else's real, smaller-
        # magnitude variation into a sliver of the colorbar near zero --
        # same category of bug as the earlier picp_field colorbar issue.
        # A robust (99th percentile) upper bound clips rare outliers instead
        # of letting them set the scale for the whole field.
        off_vmax = max(float(np.percentile(np.concatenate([down_offset.ravel(), up_offset.ravel()]), 99)), 1e-12)
        im_down = ax_down_off.imshow(down_offset, cmap='magma', vmin=0, vmax=off_vmax)
        ax_down_off.set_title(f'{channel_names[c]}: mean - lower (down offset)')
        fig.colorbar(im_down, ax=ax_down_off, fraction=0.046, pad=0.04)

        im_up = ax_up_off.imshow(up_offset, cmap='magma', vmin=0, vmax=off_vmax)
        ax_up_off.set_title(f'{channel_names[c]}: upper - mean (up offset)')
        fig.colorbar(im_up, ax=ax_up_off, fraction=0.046, pad=0.04)

        im_w = ax_width.imshow(width[c], cmap='magma')
        ax_width.set_title(f'{channel_names[c]}: width (upper-lower)')
        fig.colorbar(im_w, ax=ax_width, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved prediction+bounds plot to {out_path}')


def load_and_plot(pth_path, config_path, split='train', index=0, out_path=None, dataset=None):
    """dataset: same dataset-injection seam calibrate()-style functions
    have elsewhere in this project, for validate_synthetic.py to run
    this on in-memory tensors instead of real .pfb files."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    save_dict = torch.load(pth_path, map_location='cpu')
    model_def = save_dict['model_def']
    dtype = get_dtype(config.get('dtype', 'float32'))

    # eps/bias_init are plain Python attributes on PositiveResNetWrapper,
    # not part of state_dict() -- build_networks(model_def) alone would
    # silently reconstruct net_up/net_down with ITS OWN defaults (eps=0.2)
    # regardless of what this .pth was actually trained with, corrupting
    # PositiveBias's activation floor while the conv weights still load
    # correctly via load_state_dict below (this is exactly what was
    # happening before main_pi3nn.py started saving these two keys).
    bias_init = save_dict.get('bias_init', 3.0)
    eps = save_dict.get('eps', 0.2)
    print(f'[plot_prediction] loaded bias_init={bias_init!r}, eps={eps!r} from {pth_path} '
          f'(floor=sqrt(eps)={eps**0.5:.4g})')
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
        # Same construction main_pi3nn.py's train() uses -- scaler_yaml
        # isn't a ParFlowDataset kwarg (scaling is applied via net_mean's
        # own scale_* methods below, using the scalers already baked
        # into model_def from training, not recomputed here).
        data_def = dict(config['data_def'])
        data_def.pop('scaler_yaml', None)
        valid_fraction = config.get('valid_fraction', 0.1)
        ds = ParFlowDataset(**data_def, dtype=dtype, valid_fraction=valid_fraction, split=split)

    state, evaptrans, params, target = ds[index]
    state, evaptrans, params, target = (t.unsqueeze(0) for t in (state, evaptrans, params, target))
    net_mean.scale_pressure(state)
    net_mean.scale_evaptrans(evaptrans)
    net_mean.scale_statics(params)

    with torch.no_grad():
        mean_pred_t = net_mean(state, evaptrans, params)
        up_pred_t = net_up(state, evaptrans, params)
        down_pred_t = net_down(state, evaptrans, params)

    mean_pred = mean_pred_t[0].numpy()
    up_pred = up_pred_t[0].numpy()
    down_pred = down_pred_t[0].numpy()

    # Diagnostic: is a flat-looking offset panel a plotting artifact, or
    # does the RAW network output genuinely have near-zero spatial
    # variance for this channel/sample? (std here, not the calibrated
    # offset, isolates net_up/net_down's own behavior from c_up_field's.)
    for c in range(up_pred.shape[0]):
        print(
            f'[plot_prediction] channel {c}: up_pred range [{up_pred[c].min():.4g}, {up_pred[c].max():.4g}] '
            f'(std {up_pred[c].std():.4g}), down_pred range [{down_pred[c].min():.4g}, {down_pred[c].max():.4g}] '
            f'(std {down_pred[c].std():.4g})'
        )

    if 'c_up_field' in save_dict:
        c_up = save_dict['c_up_field'].numpy()
        c_down = save_dict['c_down_field'].numpy()
    else:
        c_up = save_dict['c_up'].numpy()
        c_down = save_dict['c_down'].numpy()

    # Diagnostic: is the calibration CONSTANT itself unexpectedly large,
    # or is the offset inflation coming from somewhere else (e.g. a stale
    # checkpoint, a units mismatch)? Printed per-channel so it can be
    # directly compared against the training run's own reported mpiw.
    for c in range(c_up.shape[0]):
        print(f'[plot_prediction] channel {c}: c_up_field range [{c_up[c].min():.4g}, {c_up[c].max():.4g}], '
              f'c_down_field range [{c_down[c].min():.4g}, {c_down[c].max():.4g}]')

    # Unscale to physical pressure units for plotting -- mean_pred is a
    # point value (full affine unscale_pressure); up_pred/down_pred are
    # magnitudes with no meaningful zero point, so they need the
    # sigma-only unscale_pressure_width instead (see model.py's
    # unscale_pressure_width docstring for the derivation -- using
    # unscale_pressure on a magnitude would incorrectly shift it by the
    # per-channel mean). Mutates the still-batched torch tensors, same
    # convention as scale_pressure/scale_evaptrans/scale_statics above.
    net_mean.unscale_pressure(mean_pred_t)
    net_mean.unscale_pressure_width(up_pred_t)
    net_mean.unscale_pressure_width(down_pred_t)
    mean_pred = mean_pred_t[0].numpy()
    up_pred = up_pred_t[0].numpy()
    down_pred = down_pred_t[0].numpy()

    upper, lower, width = compute_bounds(mean_pred, up_pred, down_pred, c_up, c_down)
    channel_names = model_def.get('pressure_names')
    out_path = out_path or pth_path.rsplit('.pth', 1)[0] + f'_{split}_idx{index}_prediction.png'
    plot_prediction_fields(mean_pred, upper, lower, width, out_path, channel_names=channel_names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth', required=True, help='Path to a {name}_pi3nn.pth file')
    parser.add_argument('--config', required=True, help='The same yaml config used for training (for data_def)')
    parser.add_argument('--split', choices=['train', 'valid'], default='train')
    parser.add_argument('--index', type=int, default=0, help='Which sample in the split to plot')
    parser.add_argument('--out', default=None, help='Output PNG path (default: alongside --pth)')
    args = parser.parse_args()
    load_and_plot(args.pth, args.config, split=args.split, index=args.index, out_path=args.out)
