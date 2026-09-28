"""
Subsets a CONUS1 domain (static parameters + transient pressure/evaptrans)
from Princeton's HydroData catalog and writes it out as .pfb files in the
layout emulator-1ts/dataset.py expects: <base_dir>/<run_name>/static/*.pfb
and <base_dir>/<run_name>/transient/{pressure,evaptrans}.NNNNN.pfb

Adapted from make_subset_domain_CONUS2.1.py after CONUS2 transient pressure
data (pressure_head) turned out not to be exposed via the HydroData API for
this account -- CONUS1's conus1_baseline_mod dataset is, and has both
pressure_head and evapotranspiration.

Notable differences from the original CONUS2 emulator design, discovered
while setting this up (see git history / project notes for the full
debugging trail):
  - CONUS1 has 5 vertical pressure layers, not CONUS2's 10.
  - There's no true multi-layer, hourly "parflow_evaptrans" forcing term
    available for CONUS1 the way there was for CONUS2. The closest
    available variable is "evapotranspiration" -- a single 2D field
    (no z-dimension), daily resolution only, and a coarser aggregate
    quantity than the internal ParFlow forcing term. Pressure is pulled
    at daily resolution too here to keep it aligned with evaptrans, rather
    than hourly. This is a deliberate simplification to get a baseline
    training run working, not a physically equivalent substitute -- revisit
    if the CONUS1 baseline needs to be more rigorous later.

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
    parser.add_argument('--transient-dataset', default='conus1_baseline_mod',
                         help='Dataset to pull pressure/evaptrans from -- '
                              'conus1_baseline_mod confirmed to have both '
                              'pressure_head and evapotranspiration')
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

    # --- Pull pressure + evaptrans, aligned at daily resolution ---
    # (evapotranspiration is only available daily for conus1_baseline_mod --
    # confirmed via hf.get_catalog_entry -- so pressure is pulled daily too
    # rather than mixing resolutions)
    data_p = hf.get_gridded_data({
        'dataset': args.transient_dataset, 'variable': 'pressure_head',
        'temporal_resolution': 'daily',
        'start_time': args.transient_start, 'end_time': args.transient_end,
        'grid_bounds': ij_bounds,
    })
    print(f'Pressure downloaded, shape: {data_p.shape}')
    print(f'(shape[1] here is the number of vertical layers for CONUS1)')

    data_et = hf.get_gridded_data({
        'dataset': args.transient_dataset, 'variable': 'evapotranspiration',
        'temporal_resolution': 'daily',
        'start_time': args.transient_start, 'end_time': args.transient_end,
        'grid_bounds': ij_bounds,
    })
    print(f'Evaptrans downloaded, shape: {data_et.shape}')

    for t in range(data_p.shape[0]):
        write_pfb(
            file=f'{transient_write_dir}/pressure.{t:05d}.pfb',
            array=data_p[t, :, :, :], dist=False,
        )
        et_frame = data_et[t]
        if et_frame.ndim == 2:
            # evapotranspiration has no z-dimension (has_z: '' in the catalog
            # entry, unlike pressure_head) -- add a singleton layer so the
            # written file is still a 3D (1, ny, nx) array like dataset.py
            # expects (it slices evaptrans[0:n_evaptrans, :, :])
            et_frame = et_frame[None, :, :]
        write_pfb(
            file=f'{transient_write_dir}/evaptrans.{t:05d}.pfb',
            array=et_frame, dist=False,
        )
    print(f'Pressure and ET files written to {transient_write_dir}')
    print(f'Done. Run directory: {input_dir}')


if __name__ == '__main__':
    main()
