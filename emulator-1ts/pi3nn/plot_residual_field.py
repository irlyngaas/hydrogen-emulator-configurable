"""
Plot the TRUE residual field for one real sample -- PI3NN's one-sided
targets (up_target = (target - mean_pred).clamp(min=0), down_target =
(mean_pred - target).clamp(min=0), both in net_mean's own SCALED space)
computed exactly as losses.residual_targets defines them, but using the
real ground-truth target instead of net_up/net_down's prediction of it.

Complements plot_prediction.py (which shows net_up/net_down's PREDICTION
of this residual): motivated by a real finding on a full-year run where
channel 4's raw up_pred/down_pred looked spatially smooth/coherent while
channels 0-3's looked speckled/noise-like. c_up_field/c_down_field can't
distinguish "real structure" from "noise" here since they're DCT-smoothed
by construction regardless of what's underneath -- this script checks
the TRUE residual's own spatial structure directly, independent of
whether net_up/net_down learned it well.

Only needs net_mean (net_up/net_down's weights/eps are irrelevant to
computing the true residual) -- loaded via get_model directly rather
than build_networks, since there's nothing here for bias_init/eps to
affect.

Usage:
  python3 -m pi3nn.plot_residual_field --pth /path/to/NAME_pi3nn.pth \
      --config /path/to/pi3nn_boxtest_config.yaml \
      --split train --index 0 [--out /path/to/output.png]
"""
import argparse

import matplotlib
matplotlib.use('Agg')  # headless (Frontier batch jobs have no display) -- must precede pyplot import
import numpy as np
import torch
import yaml

from model import get_model
from utils import get_dtype

# ParFlowDataset is imported lazily inside load_and_plot (below), not here --
# dataset.py eagerly imports xarray/xbatcher at module level, which would
# otherwise be required just to import THIS module even when a dataset is
# injected. Same reasoning as plot_prediction.py's lazy import.


def compute_residual_fields(target_scaled, mean_pred):
    """target_scaled/mean_pred: (out_channels, H, W) numpy arrays, both
    already in net_mean's own standardized units (same space net_up/
    net_down's targets are computed in during training -- see
    losses.residual_targets). Returns (up_target, down_target), each
    (out_channels, H, W), matching residual_targets' diff.clamp(min=0) /
    (-diff).clamp(min=0) exactly."""
    diff = target_scaled - mean_pred
    up_target = np.clip(diff, a_min=0, a_max=None)
    down_target = np.clip(-diff, a_min=0, a_max=None)
    return up_target, down_target


def plot_residual_fields(up_target, down_target, out_path, channel_names=None):
    """up_target/down_target: (out_channels, H, W) numpy arrays. Saves
    one PNG, one row per channel, two columns: up_target, down_target.
    Same 99th-percentile robust shared scale as plot_prediction.py's
    offset panels -- a single outlier residual pixel can otherwise
    dominate a raw max() and wash out real variation elsewhere into a
    sliver of the colorbar (same bug class caught there)."""
    import matplotlib.pyplot as plt

    out_channels = up_target.shape[0]
    channel_names = channel_names or [f'channel {c}' for c in range(out_channels)]
    fig, axes = plt.subplots(out_channels, 2, figsize=(8, 3.2 * out_channels), squeeze=False)

    for c in range(out_channels):
        ax_up, ax_down = axes[c]

        vmax = max(float(np.percentile(np.concatenate([up_target[c].ravel(), down_target[c].ravel()]), 99)), 1e-12)

        im_up = ax_up.imshow(up_target[c], cmap='magma', vmin=0, vmax=vmax)
        ax_up.set_title(f'{channel_names[c]}: true up_target (target > mean_pred)')
        fig.colorbar(im_up, ax=ax_up, fraction=0.046, pad=0.04)

        im_down = ax_down.imshow(down_target[c], cmap='magma', vmin=0, vmax=vmax)
        ax_down.set_title(f'{channel_names[c]}: true down_target (mean_pred > target)')
        fig.colorbar(im_down, ax=ax_down, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved true-residual plot to {out_path}')


def load_and_plot(pth_path, config_path, split='train', index=0, out_path=None, dataset=None):
    """dataset: same dataset-injection seam plot_prediction.py has, for
    validate_synthetic.py to run this on in-memory tensors instead of
    real .pfb files."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    save_dict = torch.load(pth_path, map_location='cpu')
    model_def = save_dict['model_def']
    dtype = get_dtype(config.get('dtype', 'float32'))

    net_mean = get_model('resnet', model_def)
    net_mean.load_state_dict(save_dict['net_mean'])
    net_mean = net_mean.to(dtype).eval()

    if dataset is not None:
        ds = dataset
    else:
        from dataset import ParFlowDataset
        data_def = dict(config['data_def'])
        data_def.pop('scaler_yaml', None)
        valid_fraction = config.get('valid_fraction', 0.1)
        ds = ParFlowDataset(**data_def, dtype=dtype, valid_fraction=valid_fraction, split=split)

    # ParFlowDataset's bgen is built with shuffle=True, so --index does NOT
    # correspond to a stable or sequential point in time -- whether it even
    # resolves to the SAME real timestep across separate process runs
    # depends on xbatcher's shuffle being deterministic, which isn't
    # verified here. Resolving and printing the real underlying file
    # removes any dependence on that assumption: it's what THIS run of the
    # script actually used, not an inference from a separate lookup.
    if hasattr(ds, 'bgen') and hasattr(ds, 'pressure_files'):
        time_index = ds.bgen[index]['time'].values[0]
        print(f'[plot_residual_field] --index {index} (split={split!r}) resolved to '
              f'time_index={time_index}, file={ds.pressure_files["t"][time_index]}')

    state, evaptrans, params, target = ds[index]
    state, evaptrans, params, target = (t.unsqueeze(0) for t in (state, evaptrans, params, target))
    target_scaled = target.clone()
    net_mean.scale_pressure(state)
    net_mean.scale_pressure(target_scaled)  # same standardized space mean_pred comes out in
    net_mean.scale_evaptrans(evaptrans)
    net_mean.scale_statics(params)

    with torch.no_grad():
        mean_pred = net_mean(state, evaptrans, params)

    mean_pred_np = mean_pred[0].numpy()
    target_scaled_np = target_scaled[0].numpy()
    up_target, down_target = compute_residual_fields(target_scaled_np, mean_pred_np)

    # Diagnostic: directly comparable to plot_prediction.py's printed
    # up_pred/down_pred stats -- is net_up/net_down's prediction tracking
    # the true residual's own scale, or diverging from it?
    for c in range(up_target.shape[0]):
        print(
            f'[plot_residual_field] channel {c}: true up_target range [{up_target[c].min():.4g}, {up_target[c].max():.4g}] '
            f'(std {up_target[c].std():.4g}), true down_target range [{down_target[c].min():.4g}, {down_target[c].max():.4g}] '
            f'(std {down_target[c].std():.4g})'
        )

    channel_names = model_def.get('pressure_names')
    out_path = out_path or pth_path.rsplit('.pth', 1)[0] + f'_{split}_idx{index}_true_residual.png'
    plot_residual_fields(up_target, down_target, out_path, channel_names=channel_names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth', required=True, help='Path to a {name}_pi3nn.pth file')
    parser.add_argument('--config', required=True, help='The same yaml config used for training (for data_def)')
    parser.add_argument('--split', choices=['train', 'valid'], default='train')
    parser.add_argument('--index', type=int, default=0, help='Which sample in the split to plot')
    parser.add_argument('--out', default=None, help='Output PNG path (default: alongside --pth)')
    args = parser.parse_args()
    load_and_plot(args.pth, args.config, split=args.split, index=args.index, out_path=args.out)
