"""
Quantify picp_field's edge-band vs. interior reliability directly,
rather than relying on eyeballing a noisy heatmap -- plot_spatial_field.py's
picp_field panel didn't show an obviously edge-concentrated pattern at
overlap=4, because (per the patch-tiling arithmetic worked out
elsewhere in this project) most of the domain is single-covered
regardless of edge/interior at a small overlap -- there isn't much
edge/interior CONTRAST yet to see visually. This gives a precise number
now, and the same metric to compare against after an overlap bump (e.g.
overlap=8, which fixes interior double-coverage while leaving the edge
band -- width patch_size-overlap -- still single-covered).

'absolute'-mode only: the edge band is a real position in the fixed
CONUS1 domain grid, which patch_relative mode's position-within-patch
indexing doesn't have.

Usage:
  python3 -m pi3nn.edge_vs_interior_picp --pth /path/to/NAME_pi3nn.pth \
      --split train --patch-size 16 --overlap 4
"""
import argparse

import numpy as np
import torch


def edge_interior_masks(H, W, band_width):
    """Boolean (H, W) masks: edge = within band_width of ANY domain
    border (the single-covered 'pure core' width worked out for an edge
    patch, patch_size - overlap), interior = everything else."""
    edge = np.zeros((H, W), dtype=bool)
    edge[:band_width, :] = True
    edge[-band_width:, :] = True
    edge[:, :band_width] = True
    edge[:, -band_width:] = True
    return edge, ~edge


def summarize(picp_field, edge_mask, interior_mask, channel_names=None):
    """picp_field: (out_channels, H, W), may contain NaN (unvisited
    cells). Prints mean/min picp for the edge band vs. the interior,
    per channel -- the direct, quantitative version of what the
    picp_field heatmap only shows qualitatively (and noisily)."""
    out_channels = picp_field.shape[0]
    channel_names = channel_names or [f'channel {c}' for c in range(out_channels)]
    for c in range(out_channels):
        field = picp_field[c]
        edge_vals = field[edge_mask]
        interior_vals = field[interior_mask]
        edge_vals = edge_vals[~np.isnan(edge_vals)]
        interior_vals = interior_vals[~np.isnan(interior_vals)]
        print(
            f'{channel_names[c]}: edge picp mean={edge_vals.mean():.4f} min={edge_vals.min():.4f} '
            f'(n={edge_vals.size}) | interior picp mean={interior_vals.mean():.4f} min={interior_vals.min():.4f} '
            f'(n={interior_vals.size})'
        )


def load_and_summarize(pth_path, split, patch_size, overlap):
    save_dict = torch.load(pth_path, map_location='cpu')
    if 'picp_field' not in save_dict['results'][split]:
        raise ValueError(
            f"{pth_path}'s {split!r} results have no 'picp_field' -- saved before "
            f"the per-cell PICP fix, or calibration_mode != 'spatial_field'"
        )
    picp_field = save_dict['results'][split]['picp_field'].numpy()
    H, W = picp_field.shape[1:]
    band_width = patch_size - overlap
    edge_mask, interior_mask = edge_interior_masks(H, W, band_width)
    channel_names = save_dict['model_def'].get('pressure_names')
    print(f'domain {H}x{W}, edge band width={band_width} (patch_size={patch_size} - overlap={overlap}), '
          f'{edge_mask.sum()} edge cells, {interior_mask.sum()} interior cells')
    summarize(picp_field, edge_mask, interior_mask, channel_names=channel_names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth', required=True, help='Path to a {name}_pi3nn.pth file')
    parser.add_argument('--split', choices=['train', 'valid'], default='train')
    parser.add_argument('--patch-size', type=int, default=16, help='Must match the run\'s data_def.patch_size')
    parser.add_argument('--overlap', type=int, default=4, help='Must match the run\'s data_def.overlap')
    args = parser.parse_args()
    load_and_summarize(args.pth, args.split, args.patch_size, args.overlap)
