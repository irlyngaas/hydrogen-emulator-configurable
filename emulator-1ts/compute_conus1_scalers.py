"""
Compute normalization scalers (mean/std) directly from a CONUS1 subset
pulled by notebooks/make_subset_domain_CONUS1.py, and print the
per-parameter layer counts / in_channels needed to configure a training
config for that data.

This computes scalers from whatever data you actually downloaded, not a
full-year, subsampled calculation like CONUS2_Data_Prep/ does for CONUS2 --
appropriate for a first baseline/smoke-test run, not a scientifically
calibrated normalization. Recompute if you significantly grow the dataset.

Usage:
    python compute_conus1_scalers.py --run-dir /path/to/CONUS1_boxtest_.../
"""
import argparse
import glob
import os

import numpy as np
import yaml
from parflow.tools.io import read_pfb


def safe_stats(values):
    mean = float(np.nanmean(values))
    std = float(np.nanstd(values))
    if std < 1e-15:
        std = 1.0
    return mean, std


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True,
                         help='Run directory containing static/ and transient/ subfolders')
    parser.add_argument('--out', default='conus1_scalers.yaml')
    args = parser.parse_args()

    static_dir = os.path.join(args.run_dir, 'static')
    transient_dir = os.path.join(args.run_dir, 'transient')

    scalers = {}
    param_layer_counts = {}

    print('--- Static parameters ---')
    static_files = sorted(glob.glob(os.path.join(static_dir, '*.pfb')))
    for f in static_files:
        name = os.path.splitext(os.path.basename(f))[0]
        data = read_pfb(f)
        nlayers = data.shape[0]
        param_layer_counts[name] = nlayers
        print(f'{name}: shape {data.shape} ({nlayers} layer(s))')
        if nlayers == 1:
            mean, std = safe_stats(data)
            scalers[name] = {'mean': mean, 'std': std}
        else:
            for i in range(nlayers):
                mean, std = safe_stats(data[i])
                scalers[f'{name}_{i}'] = {'mean': mean, 'std': std}

    print('--- Pressure ---')
    pressure_files = sorted(glob.glob(os.path.join(transient_dir, 'pressure.*.pfb')))
    pressure_stack = np.stack([read_pfb(f) for f in pressure_files])  # (T, nz, ny, nx)
    n_pressure_layers = pressure_stack.shape[1]
    print(f'{len(pressure_files)} files, {n_pressure_layers} layer(s)')
    for i in range(n_pressure_layers):
        mean, std = safe_stats(pressure_stack[:, i])
        # keyed press_diff_i to match model.py's scale_pressure lookup, even
        # though these stats come from raw pressure snapshots, not diffs --
        # same "_pressure" scaler variant convention already used elsewhere
        # in this repo (see emulator-1ts/readme.md)
        scalers[f'press_diff_{i}'] = {'mean': mean, 'std': std}

    print('--- Evaptrans ---')
    evaptrans_files = sorted(glob.glob(os.path.join(transient_dir, 'evaptrans.*.pfb')))
    evaptrans_stack = np.stack([read_pfb(f) for f in evaptrans_files])  # (T, n_evaptrans, ny, nx)
    n_evaptrans = evaptrans_stack.shape[1]
    print(f'{len(evaptrans_files)} files, {n_evaptrans} layer(s)')
    for i in range(n_evaptrans):
        mean, std = safe_stats(evaptrans_stack[:, i])
        scalers[f'evaptrans_{i}'] = {'mean': mean, 'std': std}

    with open(args.out, 'w') as fh:
        yaml.dump(scalers, fh, sort_keys=False)
    print(f'\nWrote scalers to {args.out}')

    in_channels = sum(param_layer_counts.values()) + n_evaptrans + n_pressure_layers
    param_names = [os.path.splitext(os.path.basename(f))[0] for f in static_files]
    print('\n--- Config values for this data ---')
    print(f'parameter_list: {param_names}')
    print(f'param_nlayer: {[0] * len(static_files)}  # 0 = use all layers for each')
    print(f'n_evaptrans: {n_evaptrans}')
    print(f'out_channels: {n_pressure_layers}')
    print(f'in_channels: {in_channels}  '
          f'(= sum(static layers)={sum(param_layer_counts.values())} '
          f'+ n_evaptrans={n_evaptrans} + {n_pressure_layers} pressure layers)')


if __name__ == '__main__':
    main()
