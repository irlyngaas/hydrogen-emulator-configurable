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
    Saves one PNG, one row per channel, four columns (mean, lower,
    upper, width) -- lower before upper so reading left-to-right tracks
    the physical ordering of the interval. mean/lower/upper share one
    color scale per channel (same physical quantity, directly
    comparable); width gets its own (a different quantity, always
    >= 0)."""
    import matplotlib.pyplot as plt

    out_channels = mean_pred.shape[0]
    channel_names = channel_names or [f'channel {c}' for c in range(out_channels)]
    fig, axes = plt.subplots(out_channels, 4, figsize=(16, 3.2 * out_channels), squeeze=False)

    for c in range(out_channels):
        ax_mean, ax_lower, ax_upper, ax_width = axes[c]
        vmin = min(mean_pred[c].min(), lower[c].min(), upper[c].min())
        vmax = max(mean_pred[c].max(), lower[c].max(), upper[c].max())
        for ax, field, label in (
            (ax_mean, mean_pred[c], 'mean'),
            (ax_lower, lower[c], 'lower'),
            (ax_upper, upper[c], 'upper'),
        ):
            im = ax.imshow(field, cmap='viridis', vmin=vmin, vmax=vmax)
            ax.set_title(f'{channel_names[c]}: {label}')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

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

    net_mean, net_up, net_down = build_networks(model_def)
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
        mean_pred = net_mean(state, evaptrans, params)
        up_pred = net_up(state, evaptrans, params)
        down_pred = net_down(state, evaptrans, params)

    mean_pred = mean_pred[0].numpy()
    up_pred = up_pred[0].numpy()
    down_pred = down_pred[0].numpy()

    if 'c_up_field' in save_dict:
        c_up = save_dict['c_up_field'].numpy()
        c_down = save_dict['c_down_field'].numpy()
    else:
        c_up = save_dict['c_up'].numpy()
        c_down = save_dict['c_down'].numpy()

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
