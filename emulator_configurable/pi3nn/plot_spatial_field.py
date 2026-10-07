"""
Plot spatial-field PI3NN calibration results (c_up_field/c_down_field/
picp_field) as per-channel heatmaps -- the diagnostic this whole feature
exists for: a pooled PICP number can read as a clean 0.95 everywhere
while hiding real spatial heterogeneity (see picp_spatial_std/min/max in
the calibration output). This lets you SEE where that heterogeneity
actually is instead of just reading it off summary statistics.

plot_calibration_fields (the actual drawing logic) is format-agnostic
(plain numpy arrays in, a saved PNG out) and is vendored identically in
emulator-1ts/pi3nn/plot_spatial_field.py -- same vendoring rationale as
boundary_optimizer.py (the two packages can't import across each other).
Only the loader differs, since calibrate_spatial_field() saves JSON here
but emulator-1ts's main_pi3nn.py saves a .pth dict.

Usage:
  python -m emulator_configurable.pi3nn.plot_spatial_field \
      --json /path/to/EXPERIMENT_MODE_COORDS_spatial_calibration.json \
      [--quantile 0.9] [--out /path/to/output.png]
"""
import argparse
import json

import matplotlib
matplotlib.use('Agg')  # headless (Frontier batch jobs have no display) -- must precede pyplot import
import numpy as np


def plot_calibration_fields(c_up_field, c_down_field, picp_field, out_path, quantile=None, channel_names=None):
    """c_up_field/c_down_field/picp_field: numpy arrays, shape
    (out_channels, H, W) (picp_field may contain NaN where a cell had no
    data -- only possible in 'absolute' coordinate mode). quantile: the
    target PICP (e.g. 0.95), used to center picp_field's colormap on the
    target rather than an arbitrary midpoint, so under/over-coverage is
    visually obvious at a glance rather than just a shade of one hue.
    Saves one PNG, one row per channel, three columns (c_up, c_down,
    picp)."""
    import matplotlib.pyplot as plt

    out_channels = c_up_field.shape[0]
    channel_names = channel_names or [f'channel {c}' for c in range(out_channels)]
    fig, axes = plt.subplots(out_channels, 3, figsize=(12, 3.2 * out_channels), squeeze=False)

    picp_center = quantile if quantile is not None else float(np.nanmean(picp_field))

    for c in range(out_channels):
        ax_up, ax_down, ax_picp = axes[c]

        im_up = ax_up.imshow(c_up_field[c], cmap='viridis')
        ax_up.set_title(f'{channel_names[c]}: c_up_field')
        fig.colorbar(im_up, ax=ax_up, fraction=0.046, pad=0.04)

        im_down = ax_down.imshow(c_down_field[c], cmap='viridis')
        ax_down.set_title(f'{channel_names[c]}: c_down_field')
        fig.colorbar(im_down, ax=ax_down, fraction=0.046, pad=0.04)

        # Diverging colormap centered on the target quantile (or the
        # field's own mean if no target given) -- makes under/over-
        # coverage visually obvious instead of varying shades of one hue.
        # TwoSlopeNorm (not a symmetric vmin/vmax around the center): PICP
        # is physically bounded in [0, 1] and its distribution around the
        # target is usually skewed (e.g. mostly undercovering, rarely
        # overcovering past the target) -- a symmetric range around the
        # center would pad the color scale out past 1.0 on the side with
        # less deviation, wasting most of the dynamic range on values that
        # don't exist. TwoSlopeNorm instead maps the center to white and
        # each side's ACTUAL min/max to full color, independently.
        from matplotlib.colors import TwoSlopeNorm
        chan_field = picp_field[c]
        vmin = float(np.nanmin(chan_field))
        vmax = float(np.nanmax(chan_field))
        if vmin == vmax:
            vmin, vmax = vmin - 1e-6, vmax + 1e-6
        # TwoSlopeNorm requires vmin < vcenter < vmax; if every cell is on
        # one side of the target (e.g. entirely undercovering), nudge the
        # center to just inside the data range rather than crash.
        center = min(max(picp_center, vmin + 1e-9), vmax - 1e-9)
        norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
        im_picp = ax_picp.imshow(chan_field, cmap='RdBu_r', norm=norm)
        title = f'{channel_names[c]}: picp_field'
        if quantile is not None:
            title += f' (target {quantile})'
        ax_picp.set_title(title)
        fig.colorbar(im_picp, ax=ax_picp, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved spatial-field plot to {out_path}')


def load_and_plot_json(json_path, out_path=None, quantile=None):
    """Loads a calibrate_spatial_field() JSON result and plots it.
    quantile defaults to None (colormap centered on the field's own
    mean) -- pass the config's own 'quantile' value explicitly for the
    target-centered colormap."""
    with open(json_path) as f:
        results = json.load(f)
    channel_keys = sorted(results.keys(), key=lambda k: int(k.split('_')[1]))
    if 'picp_field' not in results[channel_keys[0]]:
        raise ValueError(
            f"{json_path} has no 'picp_field' key -- it was saved before that was added to "
            f'calibrate_spatial_field(), or calibration_mode was not spatial_field for this run.'
        )
    c_up_field = np.stack([np.array(results[k]['c_up_field']) for k in channel_keys])
    c_down_field = np.stack([np.array(results[k]['c_down_field']) for k in channel_keys])
    picp_field = np.stack([np.array(results[k]['picp_field']) for k in channel_keys])

    out_path = out_path or json_path.rsplit('.json', 1)[0] + '_fields.png'
    plot_calibration_fields(c_up_field, c_down_field, picp_field, out_path, quantile=quantile, channel_names=channel_keys)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', required=True, help='Path to a *_spatial_calibration.json file')
    parser.add_argument('--out', default=None, help='Output PNG path (default: alongside --json)')
    parser.add_argument('--quantile', type=float, default=None, help="The run's target quantile, for a target-centered picp_field colormap")
    args = parser.parse_args()
    load_and_plot_json(args.json, out_path=args.out, quantile=args.quantile)
