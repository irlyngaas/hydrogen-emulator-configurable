"""
Local .pfb-based sequence dataset for ForcedSTRNN-style multi-timestep
training, reading the same layout notebooks/make_subset_domain_CONUS1.py
produces: <data_dir>/<run_name>/static/*.pfb and
<data_dir>/<run_name>/transient/{pressure,evaptrans}.NNNNN.pfb

This exists instead of using data_loader.py's zarr-based pipeline because:
- the zarr archives this repo's examples point at (e.g. the top-level
  README's conus1_2005_preprocessed.zarr) look like a pre-built HydroFrame
  product living on Princeton's /hydrodata filesystem, not something any
  script in this repo builds -- and may not be reachable via the general
  hf_hydrodata API the same way CONUS2 transient pressure wasn't.
- data_loader.py also depends on torchdata.datapipes, an API deprecated/
  removed in current torchdata releases.

Unlike emulator-1ts/dataset.py (which this closely mirrors), __getitem__
returns a *sequence* of consecutive timesteps rather than a single (t, t+1)
pair, matching what ForcedSTRNN.forward expects:
  forcings:      (sequence_length, n_forcing, patch, patch)
  init_cond:     (1, n_state, patch, patch)      -- state at the window's t=0
  static_inputs: (1, n_static, patch, patch)
  target:        (sequence_length, n_state, patch, patch) -- state at t=1..T

ForcedSTRNN has no built-in scale_pressure/scale_statics/scale_evaptrans the
way emulator-1ts's ResNet does -- data_loader.py's pipe normally applies
scaling upstream before tensors are built, so this dataset applies it itself
in __getitem__ instead, given a scalers dict in the same
{name: StandardScaler(...)} form emulator_configurable/scalers.py builds.
"""
import numpy as np
import torch
import xbatcher as xb
import xarray as xr

from glob import glob
from parflow.tools.io import read_pfb
from torch.utils.data import Dataset


class ParFlowSequenceDataset(Dataset):

    def __init__(
        self, data_dir, run_name,
        parameter_list, patch_size, overlap,
        param_nlayer, sequence_length, n_evaptrans=0,
        scalers=None, dtype=torch.float32,
    ):
        super().__init__()
        self.base_dir = f'{data_dir}/{run_name}'
        self.parameter_list = parameter_list
        self.param_nlayer = param_nlayer
        self.patch_size = patch_size
        self.overlap = overlap
        self.n_evaptrans = n_evaptrans
        self.sequence_length = sequence_length
        self.scalers = scalers or {}
        self.dtype = dtype

        self.pressure_files = sorted(glob(f'{self.base_dir}/transient/pressure*.pfb'))
        self.evaptrans_files = sorted(glob(f'{self.base_dir}/transient/evaptrans*.pfb'))
        if len(self.pressure_files) < sequence_length + 1:
            raise ValueError(
                f'Need at least sequence_length + 1 ({sequence_length + 1}) '
                f'consecutive pressure timesteps to form one sequence sample, '
                f'found {len(self.pressure_files)}. Pull a wider date range '
                f'with make_subset_domain_CONUS1.py.'
            )

        size_test = read_pfb(self.pressure_files[0])
        self.Z_EXTENT = size_test.shape[0]
        self.Y_EXTENT = size_test.shape[1]
        self.X_EXTENT = size_test.shape[2]
        # Number of valid window-start positions: each sample consumes
        # sequence_length + 1 consecutive files (1 initial condition +
        # sequence_length target/forcing steps)
        self.T_EXTENT = len(self.pressure_files) - sequence_length

        self.dummy_data = xr.Dataset().assign_coords({
            'time': np.arange(self.T_EXTENT),
            'y': np.arange(self.Y_EXTENT),
            'x': np.arange(self.X_EXTENT)
        })
        self.bgen = xb.BatchGenerator(
            self.dummy_data,
            input_dims={'x': self.patch_size, 'y': self.patch_size, 'time': 1},
            input_overlap={'x': self.overlap, 'y': self.overlap},
            return_partial=False,
            shuffle=True,
        )

        self.generate_namelist()

    def generate_namelist(self):
        self.PRESSURE_NAMES = [f'press_diff_{i}' for i in range(self.Z_EXTENT)]
        self.EVAPTRANS_NAMES = [f'evaptrans_{i}' for i in range(max(self.n_evaptrans, 1))]
        self.PARAM_NAMES = []

        patch_keys = {'x': {'start': 0, 'stop': 2}, 'y': {'start': 0, 'stop': 2}}
        for (parameter, n_lay) in zip(self.parameter_list, self.param_nlayer):
            file_name = f'{self.base_dir}/static/{parameter}.pfb'
            param_temp = read_pfb(file_name, keys=patch_keys)
            if param_temp.shape[0] == 1:
                self.PARAM_NAMES.append(parameter)
            else:
                temp_namelist = [f'{parameter}_{i}' for i in range(param_temp.shape[0])]
                if n_lay > 0:
                    temp_namelist = temp_namelist[0:n_lay]
                elif n_lay < 0:
                    temp_namelist = temp_namelist[n_lay:]
                self.PARAM_NAMES.extend(temp_namelist)

    def __len__(self):
        return len(self.bgen)

    def _scale(self, x, names):
        """x: (channels, h, w) numpy array; names: per-channel scaler keys."""
        if not self.scalers:
            return x
        x = x.copy()
        for i, name in enumerate(names):
            if name in self.scalers:
                x[i] = self.scalers[name].transform(x[i])
        return x

    def _read_param_patch(self, patch_keys):
        parameter_data = []
        for (parameter, n_lay) in zip(self.parameter_list, self.param_nlayer):
            file_name = f'{self.base_dir}/static/{parameter}.pfb'
            param_temp = read_pfb(file_name, keys=patch_keys)
            if param_temp.shape[0] > 1:
                if n_lay > 0:
                    param_temp = param_temp[0:n_lay, :, :]
                elif n_lay < 0:
                    param_temp = param_temp[n_lay:, :, :]
            parameter_data.append(param_temp)
        static = np.concatenate(parameter_data, axis=0)
        return self._scale(static, self.PARAM_NAMES)

    def _read_evaptrans_patch(self, file_path, patch_keys):
        evaptrans = read_pfb(file_path, keys=patch_keys)
        if evaptrans.ndim == 2:
            evaptrans = evaptrans[np.newaxis, :, :]
        if self.n_evaptrans > 0:
            evaptrans = evaptrans[0:self.n_evaptrans, :, :]
        elif self.n_evaptrans < 0:
            evaptrans = evaptrans[self.n_evaptrans:, :, :]
        return self._scale(evaptrans, self.EVAPTRANS_NAMES)

    def _read_pressure_patch(self, file_path, patch_keys):
        pressure = read_pfb(file_path, keys=patch_keys)
        return self._scale(pressure, self.PRESSURE_NAMES)

    def __getitem__(self, idx):
        sample_indices = self.bgen[idx]
        time_index = sample_indices['time'].values[0]
        x_min, x_max = sample_indices['x'].values[[0, -1]]
        y_min, y_max = sample_indices['y'].values[[0, -1]]
        patch_keys = {
            'x': {'start': x_min, 'stop': x_max + 1},
            'y': {'start': y_min, 'stop': y_max + 1},
        }

        # Initial condition: pressure at the window's first timestep
        init_cond = self._read_pressure_patch(self.pressure_files[time_index], patch_keys)
        init_cond = torch.from_numpy(init_cond).to(self.dtype).unsqueeze(0)  # (1, nz, h, w)

        # Static params: time-invariant, read once per sample
        static_inputs = self._read_param_patch(patch_keys)
        static_inputs = torch.from_numpy(static_inputs).to(self.dtype).unsqueeze(0)  # (1, nstatic, h, w)

        # Sequence of forcings and targets for steps 1..sequence_length
        forcing_seq, target_seq = [], []
        for step in range(1, self.sequence_length + 1):
            et = self._read_evaptrans_patch(self.evaptrans_files[time_index + step - 1], patch_keys)
            forcing_seq.append(torch.from_numpy(et).to(self.dtype))

            target = self._read_pressure_patch(self.pressure_files[time_index + step], patch_keys)
            target_seq.append(torch.from_numpy(target).to(self.dtype))

        forcings = torch.stack(forcing_seq)   # (T, n_forcing, h, w)
        target = torch.stack(target_seq)      # (T, nz, h, w)

        return forcings, init_cond, static_inputs, target
