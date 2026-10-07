import xarray as xr
import os
import torch
# Need to use xbatcher from: https://github.com/arbennett/xbatcher/tree/develop
 # See readme for installation instrutions 
import xbatcher as xb
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from parflow.tools.io import read_pfb

from torch.utils.data import Dataset


def time_block_split(n, valid_fraction, split, valid_block_count=1):
    """Returns the sorted list of indices in [0, n) belonging to `split`
    ('train' or 'valid'). valid_block_count=1 (default) reproduces the
    original single-tail-holdout behavior exactly: one block spanning
    the whole range, earliest (1-valid_fraction) = train, latest
    valid_fraction = valid. >1 divides [0, n) into that many roughly-
    equal contiguous blocks and applies the SAME tail-holdout
    independently within each block, then pools every block's train (or
    valid) indices together -- so every block (e.g. one block per
    calendar month, valid_block_count=12) contributes to BOTH splits,
    instead of validation being one single contiguous chunk at the very
    end of the whole range. That matters for anything spanning more
    than about one season: with a single block, validation only ever
    sees whatever narrow window the tail happens to fall in (confirmed
    a real problem on a full-year CONUS1 run -- see
    pi3nn_frontier_status memory), rather than a representative sample
    across the full range the model was trained on.

    Pure index arithmetic, no I/O -- kept as a standalone function
    (not inlined in ParFlowDataset.__init__) specifically so it's
    testable without needing real .pfb files."""
    assert split in ('train', 'valid')
    block_edges = np.linspace(0, n, valid_block_count + 1).round().astype(int)
    train_idx, valid_idx = [], []
    for b in range(valid_block_count):
        lo, hi = int(block_edges[b]), int(block_edges[b + 1])
        block_n = hi - lo
        block_n_valid = max(1, int(round(block_n * valid_fraction)))
        block_n_train = block_n - block_n_valid
        train_idx.extend(range(lo, lo + block_n_train))
        valid_idx.extend(range(lo + block_n_train, hi))
    return train_idx if split == 'train' else valid_idx


class ParFlowDataset(Dataset):

    def __init__(
        self, data_dir, run_name,
        parameter_list, patch_size, overlap,
        param_nlayer, n_evaptrans=0, dtype=torch.float64,
        valid_fraction=0.0, split='train',
        return_coords=False, valid_block_count=1,
    ):
        super().__init__()
        self.base_dir = f'{data_dir}/{run_name}'
        self.parameter_list = parameter_list
        self.param_nlayer = param_nlayer #number of layers to use for each param, 0= use all, -n = n top layers, +n = n bottom layers
        self.patch_size = patch_size
        self.n_evaptrans = n_evaptrans
        self.overlap = overlap
        self.dtype = dtype
        # Additive, default-off: when True, __getitem__ also returns this
        # sample's (y_min, x_min) -- its absolute position in the fixed
        # CONUS1 domain grid (same grid/size every call, since X_EXTENT/
        # Y_EXTENT come from the first pressure file and every pressure
        # file in a run shares the same spatial grid). Needed only by
        # pi3nn/trainer.py's boundary_optimization_spatial_field() in
        # 'absolute' coordinate mode -- every other call site (training,
        # 'patch_relative' calibration) is unaffected since the default
        # keeps __getitem__'s return shape exactly as it was.
        self.return_coords = return_coords

        self.pressure_files = sorted(glob(f'{self.base_dir}/transient/pressure*.pfb'))
        self.pressure_files = {
            't': self.pressure_files[0:-1],
            't+1': self.pressure_files[1:]
        }

        # Time-based train/valid split -- see time_block_split's docstring.
        # valid_fraction=0.0 (default) is a no-op, so every existing
        # config/call site that doesn't pass these kwargs sees exactly
        # today's behavior (the full dataset, unsplit). valid_block_count=1
        # (default) reproduces the original single-tail-holdout behavior
        # exactly, so existing configs that only set valid_fraction/split
        # are unaffected unless they opt into valid_block_count > 1.
        if valid_fraction > 0:
            n = len(self.pressure_files['t'])
            idx = time_block_split(n, valid_fraction, split, valid_block_count)
            self.pressure_files = {k: [v[i] for i in idx] for k, v in self.pressure_files.items()}

        self.size_test = read_pfb(self.pressure_files['t'][0])
        self.X_EXTENT = self.size_test.shape[2] 
        self.Y_EXTENT = self.size_test.shape[1]
        self.Z_EXTENT = self.size_test.shape[0]
        self.T_EXTENT = len(self.pressure_files['t'])
      
        # Create a dummy dataset that will be used to pull indices for reading subsets of the data
        self.dummy_data = xr.Dataset().assign_coords({
            'time': np.arange(self.T_EXTENT),
            'z': np.arange(self.Z_EXTENT),
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
        """
        Generate a list of names that will be used to input to the model.
        This will be used as a way to record the order that the variables
        go into the model so that they can be scaled internally. See 
        model.scale_pressure, model.cale_evaptrans, and model.scale_statics
        for more information.
        """
        self.PRESSURE_NAMES = [f'press_diff_{i}' for i in range(self.Z_EXTENT)]
        self.EVAPTRANS_NAMES = [f'evaptrans_{i}' for i in range(self.n_evaptrans)]
        self.PARAM_NAMES = []
        self.OUTPUT_NAMES = [f'press_diff_{i}' for i in range(self.Z_EXTENT)]

        # Use a tiny key just to look up what we need
        patch_keys = {'x': {'start': 0, 'stop': 2},
                      'y': {'start': 0, 'stop': 2},}
        for (parameter, n_lay) in zip(self.parameter_list, self.param_nlayer):
            file_name=f'{self.base_dir}/static/{parameter}.pfb'

            # param_temp shape is (n_layers, y, x)
            param_temp = read_pfb(file_name, keys=patch_keys)

            if param_temp.shape[0] == 1:
                self.PARAM_NAMES.append(parameter)
            else: 
                temp_namelist = [f'{parameter}_{i}' for i in range(param_temp.shape[0])]
                #Grab the top n bottom or top layers if specified in the param_nlayer list
                #Grab the bottom n_lay layers
                if n_lay > 0:
                    temp_namelist = temp_namelist[0:n_lay]
                #Grab the top n_lay layers
                elif n_lay < 0:
                    temp_namelist = temp_namelist[n_lay:]

                # Add the new names to the list
                for t in temp_namelist:
                    self.PARAM_NAMES.append(t)
        
    def __len__(self):
        return len(self.bgen) 
    
    def __getitem__(self, idx):
        sample_indices = self.bgen[idx]

        # Pulling the indices we need
        time_index = sample_indices['time'].values[0]
        x_min, x_max = sample_indices['x'].values[[0, -1]]
        y_min, y_max = sample_indices['y'].values[[0, -1]]

        # Setting up the keys dictionary
        patch_keys = {
            'x': {'start': x_min, 'stop': x_max+1},
            'y': {'start': y_min, 'stop': y_max+1},
        }
    
        # Construct the state data and scale it:
        file_to_read = self.pressure_files['t'][time_index]
        state_data = read_pfb(file_to_read, keys=patch_keys)

        # Construct the target data and scale it:
        file_to_read_target = self.pressure_files['t+1'][time_index]
        target_data = read_pfb(file_to_read_target, keys=patch_keys)

        # Construct the parameter data and scale it:
        parameter_data = []
        for (parameter, n_lay) in zip(self.parameter_list, self.param_nlayer):
            file_name=f'{self.base_dir}/static/{parameter}.pfb'
            param_temp = read_pfb(file_name, keys=patch_keys)

            if param_temp.shape[0] > 1:
                #Grab the top n bottom or top layers if specified in the param_nlayer list
                #Grab the bottom n_lay layers
                if n_lay > 0:
                    param_temp = param_temp[0:n_lay,:,:]
                #Grab the top n_lay layers
                elif n_lay < 0:
                    param_temp = param_temp[n_lay:,:,:]

            parameter_data.append(param_temp)

        # Concatenate the parameter data together
        # End result is a dims of (n_parameters, y, x)
        parameter_data = np.concatenate(parameter_data, axis=0)

        #Construct the evaptrans data and scale it
        file_name_et=file_to_read.replace('pressure', 'evaptrans')
        evaptrans = (read_pfb(file_name_et, keys= patch_keys))
        #Grab the top n bottom or top layers if specified in the param_nlayer list
        #Grab the bottom n_lay layers
        if self.n_evaptrans > 0:
            evaptrans = evaptrans[0:self.n_evaptrans,:,:]
        #Grab the top n_lay layers
        elif self.n_evaptrans < 0:
            evaptrans = evaptrans[self.n_evaptrans:,:,:]
        
        # Convert everything to torch tensors
        state_data = torch.from_numpy(state_data).to(self.dtype)
        evaptrans = torch.from_numpy(evaptrans).to(self.dtype)
        parameter_data = torch.from_numpy(parameter_data).to(self.dtype)
        target_data = torch.from_numpy(target_data).to(self.dtype)

        if self.return_coords:
            coords = torch.tensor([y_min, x_min], dtype=torch.long)
            return state_data, evaptrans, parameter_data, target_data, coords
        return state_data, evaptrans, parameter_data, target_data