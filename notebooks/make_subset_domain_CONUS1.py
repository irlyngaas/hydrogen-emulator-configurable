"""
Subsets a CONUS1 domain (static parameters + transient pressure/evaptrans)
from Princeton's HydroData catalog and writes it out as .pfb files in the
layout emulator-1ts/dataset.py expects: <base_dir>/<run_name>/static/*.pfb
and <base_dir>/<run_name>/transient/{pressure,evaptrans}.NNNNN.pfb

Adapted from make_subset_domain_CONUS2.1.py after CONUS2 transient pressure
data (pressure_head) turned out not to be exposed via the HydroData API for
this account -- CONUS1 is, via the conus1_baseline_85 / conus1_baseline_mod
datasets. See the CONUS1 vs CONUS2 discussion in the accompanying repo
notes: different vertical layer count, different grid indexing than CONUS2.

You need a HydroData account: register at https://hydrogen.princeton.edu/pin
Provide credentials via --email/--pin, or the HYDRODATA_EMAIL/HYDRODATA_PIN
environment variables (preferred for non-interactive/batch runs).
"""
import argparse
import os

import hf_hydrodata as hf
import subsettools as st
from parflow.tools.io import write_pfb
from parflow.tools.fs import mkdir


DEFAULT_STATIC_VARS = [
    'slope_x', 'slope_y', 'pme', 'ss_pressure_head', 'pf_indicator',
    'porosity', 'permeability', 'van_genuchten_alpha', 'van_genuchten_n',
]
# Verified against hf.get_variables({"dataset": "conus1_domain", "grid": "conus1"}).
# Not available for conus1_domain at all (present for conus2_domain, no CONUS1
# equivalent found): pf_flowbarrier, mannings, specific_storage, sres, ssat,
# top_patch. permeability is a single field here, not split into x/y/z.


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--email', default=os.environ.get('HYDRODATA_EMAIL'),
                         help='HydroData account email (or set HYDRODATA_EMAIL)')
    parser.add_argument('--pin', default=os.environ.get('HYDRODATA_PIN'),
                         help='HydroData PIN (or set HYDRODATA_PIN)')

    parser.add_argument('--run-name', default='CONUS1_boxtest')
    parser.add_argument('--base-dir', required=True,
                         help='Directory the subset run folder will be created under')

    parser.add_argument('--grid', default='conus1')
    parser.add_argument('--static-dataset', default='conus1_domain',
                         help='Dataset to pull static variables from')
    parser.add_argument('--transient-dataset', default='conus1_baseline_85',
                         help='Dataset to pull pressure/evaptrans from '
                              '(conus1_baseline_85 or conus1_baseline_mod)')
    parser.add_argument('--static-vars', default=','.join(DEFAULT_STATIC_VARS),
                         help='Comma-separated list of static variables to subset')

    parser.add_argument('--static-start', default='2002-10-01',
                         help='Start date used to name the output folder')
    parser.add_argument('--static-end', default='2002-10-05',
                         help='End date used to name the output folder')

    parser.add_argument('--transient-start', default='2002-10-01')
    parser.add_argument('--transient-end', default='2002-10-03')

    parser.add_argument('--lower-left-i', type=int, default=1000,
                         help='CONUS1 has its own grid indexing -- do not assume '
                              'a CONUS2 box lines up the same way; check the '
                              'mask-coverage printout on first run')
    parser.add_argument('--lower-left-j', type=int, default=1000)
    parser.add_argument('--box-nx', type=int, default=63)
    parser.add_argument('--box-ny', type=int, default=67)

    return parser.parse_args()


def main():
    args = parse_args()
    if not args.email or not args.pin:
        raise SystemExit(
            'HydroData credentials required: pass --email/--pin or set '
            'HYDRODATA_EMAIL/HYDRODATA_PIN.'
        )

    print(f'Registering {args.email} for HydroData download')
    hf.register_api_pin(args.email, args.pin)

    # --- Set up output directories ---
    input_dir = os.path.join(
        args.base_dir,
        f'{args.run_name}_{args.transient_dataset}_{args.static_start}',
    )
    static_write_dir = os.path.join(input_dir, 'static')
    transient_write_dir = os.path.join(input_dir, 'transient')
    mkdir(static_write_dir)
    mkdir(transient_write_dir)

    # --- Define the domain box ---
    # Note: conus1_domain has no "mask" variable (confirmed via
    # hf.get_variables), unlike conus2_domain, so there's no equivalent
    # sanity check available here before pulling data.
    ij_bounds = (
        args.lower_left_i,
        args.lower_left_j,
        args.lower_left_i + args.box_nx,
        args.lower_left_j + args.box_ny,
    )
    ni = ij_bounds[2] - ij_bounds[0]
    nj = ij_bounds[3] - ij_bounds[1]
    print(f'bounding box: {ij_bounds}')
    print(f'ni: {ni}, nj: {nj}')

    # --- Subset static parameters ---
    variable_list = args.static_vars.split(',')
    st.subset_static(
        ij_bounds, dataset=args.static_dataset,
        write_dir=static_write_dir, var_list=variable_list,
    )
    print(f'Static variables written to {static_write_dir}')

    # --- Pull pressure + evaptrans and write as hourly .pfb files ---
    data_p = hf.get_gridded_data({
        'dataset': args.transient_dataset, 'variable': 'pressure_head',
        'temporal_resolution': 'hourly',
        'start_time': args.transient_start, 'end_time': args.transient_end,
        'grid_bounds': ij_bounds,
    })
    print(f'Pressure downloaded, shape: {data_p.shape}')
    print(f'(shape[1] here is the number of vertical layers for CONUS1)')

    data_et = hf.get_gridded_data({
        'dataset': args.transient_dataset, 'variable': 'parflow_evaptrans',
        'temporal_resolution': 'hourly',
        'start_time': args.transient_start, 'end_time': args.transient_end,
        'grid_bounds': ij_bounds,
    })
    print(f'Evaptrans downloaded, shape: {data_et.shape}')

    for hour in range(data_p.shape[0]):
        write_pfb(
            file=f'{transient_write_dir}/pressure.{hour:05d}.pfb',
            array=data_p[hour, :, :, :], dist=False,
        )
        write_pfb(
            file=f'{transient_write_dir}/evaptrans.{hour:05d}.pfb',
            array=data_et[hour, :, :, :], dist=False,
        )
    print(f'Pressure and ET files written to {transient_write_dir}')
    print(f'Done. Run directory: {input_dir}')


if __name__ == '__main__':
    main()
