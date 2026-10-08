"""
Plot one real input sample's mean prediction plus calibrated upper/
lower bounds and interval width, as per-channel heatmap panels --
complements plot_spatial_field.py (which plots the calibration
CONSTANTS: c_up_field/c_down_field/picp_field) by showing what those
constants actually produce for a real forecast. See
emulator-1ts/pi3nn/plot_prediction.py's module docstring for why this
is side-by-side heatmaps, not per-pixel error bars (vendored
identically here, see that file for the shared drawing logic's
rationale).

Works with either calibration_mode's saved output: pass a scalar
calibration JSON (calibrate()'s output, 'c_up'/'c_down' per channel) or
a spatial calibration JSON (calibrate_spatial_field()'s output,
'c_up_field'/'c_down_field' per channel) via --calibration-json;
compute_bounds broadcasts either shape correctly.

Samples here carry a sequence_length T axis (unlike emulator-1ts's
single-step samples) -- defaults to plotting the LAST rollout step
(the furthest-out, hardest prediction); pass --timestep to pick a
different one.

Usage:
  python -m emulator_configurable.pi3nn.plot_prediction \
      --config /path/to/conus1_boxtest_pi3nn_config.json \
      --up-down-mode stateless_output \
      --mean-ckpt /path/to/..._mean.ckpt \
      --up-ckpt /path/to/..._up.ckpt --down-ckpt /path/to/..._down.ckpt \
      --calibration-json /path/to/..._spatial_calibration.json \
      --index 0 [--timestep -1] [--out /path/to/output.png]
"""
import argparse
import json

import matplotlib
matplotlib.use('Agg')  # headless (Frontier batch jobs have no display) -- must precede pyplot import
import numpy as np
import torch

from ..pfb_dataset import ParFlowSequenceDataset
from ..scalers import create_scalers_from_yaml
from .calibration import _load_mean, _load_updown
from .networks import predict_updown


def compute_bounds(mean_pred, up_pred, down_pred, c_up, c_down):
    """mean_pred/up_pred/down_pred: (out_channels, H, W) numpy arrays
    for ONE sample at ONE timestep. c_up/c_down: either a per-channel
    scalar (shape (out_channels,)) or a per-channel field (shape
    (out_channels, H, W)) -- broadcasts correctly either way. Returns
    (upper, lower, width), each (out_channels, H, W)."""
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


def load_calibration_constants(calibration_json_path, out_channel):
    with open(calibration_json_path) as f:
        results = json.load(f)
    channel_keys = sorted(results.keys(), key=lambda k: int(k.split('_')[1]))
    assert len(channel_keys) == out_channel, (
        f'{calibration_json_path} has {len(channel_keys)} channels, expected {out_channel}'
    )
    if 'c_up_field' in results[channel_keys[0]]:
        c_up = np.stack([np.array(results[k]['c_up_field']) for k in channel_keys])
        c_down = np.stack([np.array(results[k]['c_down_field']) for k in channel_keys])
    else:
        c_up = np.array([results[k]['c_up'] for k in channel_keys])
        c_down = np.array([results[k]['c_down'] for k in channel_keys])
    return c_up, c_down


def load_and_plot(
    config, mean_ckpt_path, up_ckpt_path, down_ckpt_path, up_down_mode,
    calibration_json_path, index=0, timestep=-1, out_path=None, train_dataset=None,
):
    """train_dataset: same dataset-injection seam calibrate()/
    calibrate_spatial_field() have, for validate_synthetic.py to run
    this on in-memory tensors instead of real .pfb files."""
    net_mean = _load_mean('ForcedSTRNN', config['mean_model_config'], mean_ckpt_path)
    net_up = _load_updown(up_down_mode, config['updown_model_config'][up_down_mode], up_ckpt_path)
    net_down = _load_updown(up_down_mode, config['updown_model_config'][up_down_mode], down_ckpt_path)

    if train_dataset is not None:
        ds = train_dataset
    else:
        scalers = create_scalers_from_yaml(config['scaler_file']) if config.get('scaler_file') else None
        ds = ParFlowSequenceDataset(
            data_dir=config['data_dir'], run_name=config['run_name'],
            parameter_list=config['parameter_list'], patch_size=config['patch_size'],
            overlap=config['overlap'], param_nlayer=config['param_nlayer'],
            sequence_length=config['sequence_length'], n_evaptrans=config['n_evaptrans'],
            scalers=scalers, valid_fraction=config['valid_fraction'], split='train',
        )
    forcing, state, params, target = ds[index]
    forcing, state, params, target = (t.unsqueeze(0) for t in (forcing, state, params, target))

    with torch.no_grad():
        pred_up, mean_pred = predict_updown(net_mean, net_up, up_down_mode, forcing, state, params)
        pred_down, _ = predict_updown(net_mean, net_down, up_down_mode, forcing, state, params)

    t = timestep if timestep >= 0 else mean_pred.shape[1] + timestep
    mean_pred_np = mean_pred[0, t].numpy()
    up_pred_np = pred_up[0, t].numpy()
    down_pred_np = pred_down[0, t].numpy()

    # Diagnostic: is a flat-looking offset panel a plotting artifact, or
    # does the RAW network output genuinely have near-zero spatial
    # variance for this channel/sample? (std here, not the calibrated
    # offset, isolates net_up/net_down's own behavior from c_up_field's.)
    for c in range(up_pred_np.shape[0]):
        print(
            f'[plot_prediction] channel {c}: up_pred range [{up_pred_np[c].min():.4g}, {up_pred_np[c].max():.4g}] '
            f'(std {up_pred_np[c].std():.4g}), down_pred range [{down_pred_np[c].min():.4g}, {down_pred_np[c].max():.4g}] '
            f'(std {down_pred_np[c].std():.4g})'
        )

    out_channel = mean_pred_np.shape[0]
    c_up, c_down = load_calibration_constants(calibration_json_path, out_channel)
    upper, lower, width = compute_bounds(mean_pred_np, up_pred_np, down_pred_np, c_up, c_down)

    out_path = out_path or calibration_json_path.rsplit('.json', 1)[0] + f'_idx{index}_t{t}_prediction.png'
    plot_prediction_fields(mean_pred_np, upper, lower, width, out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--up-down-mode', required=True, choices=['stateless_output', 'stateless_hidden', 'recurrent_clone'])
    parser.add_argument('--mean-ckpt', required=True)
    parser.add_argument('--up-ckpt', required=True)
    parser.add_argument('--down-ckpt', required=True)
    parser.add_argument('--calibration-json', required=True)
    parser.add_argument('--index', type=int, default=0, help='Which sample in the dataset to plot')
    parser.add_argument('--timestep', type=int, default=-1, help='Which rollout step to plot (-1 = last)')
    parser.add_argument('--out', default=None, help='Output PNG path (default: alongside --calibration-json)')
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    load_and_plot(
        config, args.mean_ckpt, args.up_ckpt, args.down_ckpt, args.up_down_mode,
        args.calibration_json, index=args.index, timestep=args.timestep, out_path=args.out,
    )
