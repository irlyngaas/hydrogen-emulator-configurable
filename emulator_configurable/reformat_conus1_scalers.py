"""
Reformats the mean/std scaler YAML already produced for emulator-1ts
(emulator-1ts/compute_conus1_scalers.py) into the format
emulator_configurable/scalers.py's create_scalers_from_yaml expects: each
entry tagged with a `type` (StandardScaler), rather than bare mean/std.

Same underlying values, just a different schema for a different consumer --
no data is re-read or recomputed here.

Usage:
    python reformat_conus1_scalers.py \
        --in /path/to/conus1_scalers.yaml \
        --out /path/to/conus1_scalers_forcedstrnn.yaml
"""
import argparse
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--in', dest='in_path', required=True)
    parser.add_argument('--out', dest='out_path', required=True)
    args = parser.parse_args()

    with open(args.in_path) as f:
        scalers = yaml.load(f, Loader=yaml.FullLoader)

    reformatted = {
        name: {'type': 'StandardScaler', 'mean': values['mean'], 'std': values['std']}
        for name, values in scalers.items()
    }

    with open(args.out_path, 'w') as f:
        yaml.dump(reformatted, f, sort_keys=False)
    print(f'Wrote {len(reformatted)} reformatted scaler entries to {args.out_path}')


if __name__ == '__main__':
    main()
