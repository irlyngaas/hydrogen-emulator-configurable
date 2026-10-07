"""
Plot spatial-field PI3NN calibration results (c_up_field/c_down_field/
picp_field) as per-channel heatmaps -- the diagnostic this whole feature
exists for: a pooled PICP number can read as a clean 0.95 everywhere
while hiding real spatial heterogeneity (see picp_spatial_std/min/max in
evaluate_spatial_field()'s output). This lets you SEE where that
heterogeneity actually is instead of just reading it off summary
statistics.

plot_calibration_fields (the actual drawing logic) is format-agnostic
(plain numpy arrays in, a saved PNG out) and is vendored identically in
emulator_configurable/pi3nn/plot_spatial_field.py -- same vendoring
rationale as boundary_optimizer.py (the two packages can't import across
each other). Only the loader differs, since main_pi3nn.py saves a .pth
dict here but calibrate_spatial_field() saves JSON there.

Usage:
  python3 -m pi3nn.plot_spatial_field --pth /path/to/NAME_pi3nn.pth \
      --split train [--quantile 0.95] [--out /path/to/output.png]
"""
import argparse

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


def load_and_plot_pth(pth_path, split='train', out_path=None, quantile=None):
    """Loads a main_pi3nn.py '{name}_pi3nn.pth' result (calibration_mode
    'spatial_field') and plots it. split: 'train' or 'valid' -- which of
    evaluate_spatial_field()'s results to pull picp_field from (c_up_
    field/c_down_field are split-independent, same field either way).
    quantile defaults to None (colormap centered on the field's own
    mean) -- pass the run's own pi3nn_configs['quantile'] explicitly for
    the target-centered colormap."""
    import torch
    save_dict = torch.load(pth_path, map_location='cpu')
    if 'c_up_field' not in save_dict:
        raise ValueError(
            f"{pth_path} has no 'c_up_field' key -- it was saved with calibration_mode='scalar' "
            f"(use c_up/c_down directly, there's no spatial field to plot), or predates this feature."
        )
    if split not in save_dict['results']:
        raise ValueError(f"split={split!r} not in this file's results (has: {list(save_dict['results'].keys())})")
    if 'picp_field' not in save_dict['results'][split]:
        raise ValueError(
            f"{pth_path}'s {split!r} results have no 'picp_field' -- saved before that was "
            f'wired into evaluate_spatial_field().'
        )

    c_up_field = save_dict['c_up_field'].numpy()
    c_down_field = save_dict['c_down_field'].numpy()
    picp_field = save_dict['results'][split]['picp_field'].numpy()
    channel_names = save_dict.get('model_def', {}).get('pressure_names')

    out_path = out_path or pth_path.rsplit('.pth', 1)[0] + f'_{split}_fields.png'
    plot_calibration_fields(c_up_field, c_down_field, picp_field, out_path, quantile=quantile, channel_names=channel_names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth', required=True, help='Path to a {name}_pi3nn.pth file')
    parser.add_argument('--split', choices=['train', 'valid'], default='train')
    parser.add_argument('--out', default=None, help='Output PNG path (default: alongside --pth)')
    parser.add_argument('--quantile', type=float, default=None, help="The run's target quantile, for a target-centered picp_field colormap")
    args = parser.parse_args()
    load_and_plot_pth(args.pth, split=args.split, out_path=args.out, quantile=args.quantile)
