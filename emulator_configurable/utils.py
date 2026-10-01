import dask
import torch
import torch.nn.functional as F

from tqdm.autonotebook import tqdm
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import TQDMProgressBar


dask.config.set(**{'array.slicing.split_large_chunks': True})

def update_config_for_inference(
        config,
        inference_dataset_files,
        save_path,
        selectors
):
    # Update the config to reflect the inference settings
    config['inference_dataset_files'] = inference_dataset_files
    config['save_path'] = save_path
    config['selectors'] = selectors
    config['states'] = config.pop('targets')

    stuff_we_dont_need_for_inference = [
        'sequence_length',
        'precision',
        'logging_frequency',
        'learning_rate',
        'batch_size',
        'num_workers',
        'num_epochs',
        'gradient_loss_penalty',
        'patch_size'
    ]
    for key in stuff_we_dont_need_for_inference:
        config.pop(key, None)
    return config

def save_predictions(ds, save_path):
    if save_path.endswith('zarr'):
        ds.to_zarr(save_path, consolidated=True)
    elif save_path.endswith('nc'):
        ds.to_netcdf(save_path)

def maybe_split_3d_vars(ds):
    """
    Splits 3D variables in the given dataset along the 'z' dimension.

    This function iterates over the variables in the dataset and checks if any of them have a dimension named 'z'.
    If a variable has a 'z' dimension, it splits the variable into multiple variables along the 'z' dimension.
    The new variables are named by appending the index of the 'z' dimension to the original variable name.

    Parameters:
    - ds (xarray.Dataset): The dataset containing the variables to be split.

    Returns:
    - ds (xarray.Dataset): The dataset with the 3D variables split along the 'z' dimension.

    Example:
    >>> ds = xr.Dataset({'pressure': (['z', 'y', 'x'], np.random.rand(10, 10, 5))})
    >>> ds = maybe_split_3d_vars(ds)
    >>> print(ds)
    <xarray.Dataset>
    Dimensions:        (x: 10, y: 10, z: 5)
    Coordinates:
      * z              (z) int64 0 1 2 3 4
      * y              (y) int64 0 1 2 3 4 5 6 7 8 9
      * x              (x) int64 0 1 2 3 4 5 6 7 8 9
    Data variables:
        pressure   (z, y, x) float64 ...
        pressure_0    (x, y) float64 ...
        pressure_1    (x, y) float64 ...
        pressure_2    (x, y) float64 ...
        pressure_3    (x, y) float64 ...
        pressure_4    (x, y) float64 ...
    """
    for v in set(ds.variables) - set(ds.coords):
        if 'z' in ds[v].dims:
            for i in range(ds.sizes['z']):
                if f'{v}_{i}' not in ds:
                    ds[f'{v}_{i}'] = ds[v].isel(z=i)
    return ds

def spatial_gradient_penalty_loss(yhat, ytru, space_weight=1, loss_fun=F.mse_loss):
    """
    Calculates the spatial gradient penalty loss between predicted and true values.

    Args:
        yhat (torch.Tensor): The predicted values.
        ytru (torch.Tensor): The true values.
        space_weight (float, optional): The weight for the spatial gradient penalty. Defaults to 1.
        loss_fun (function, optional): The loss function to calculate the loss. Defaults to F.mse_loss.

    Returns:
        torch.Tensor: The calculated loss.
    """
    loss = loss_fun(ytru, yhat)

    dx_tru = torch.diff(ytru, dim=-1)
    dx_hat = torch.diff(yhat, dim=-1)
    dx_loss = loss_fun(dx_tru, dx_hat)

    dy_tru = torch.diff(ytru, dim=-2)
    dy_hat = torch.diff(yhat, dim=-2)
    dy_loss = loss_fun(dy_tru, dy_hat)

    return loss + space_weight * (dx_loss + dy_loss)


def count_parameters(model):
    """
    Returns the number of parameters in a pytorch model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sequence_to_device(seq, device):
    """
    Move a sequence of tensors to a device.
    """
    return [s.to(device) for s in seq]


def match_dims(x, target):
    """
    Reshape x to match the dimensions of target.
    """
    return x.reshape([len(x) if i == len(x) else 1 for i in target.shape])


class MetricsCallback(Callback):
    """
    PyTorch Lightning metric callback.
    """

    def __init__(self):
        super().__init__()
        self.metrics = {}

    def train_epoch_end(self, trainer, pl_module):
        for k, v in trainer.logged_metrics.items():
            if k not in self.metrics.keys():
                self.metrics[k] = [self._convert(v)]
            else:
                self.metrics[k].append(self._convert(v))

    def _convert(self, x):
        if isinstance(x, torch.Tensor):
            return x.cpu().detach().numpy()
        return x


class LitProgressBar(TQDMProgressBar):
    """
    This just avoids a bug in the progress bar for pytorch lightning
    that causes the progress bar to creep down the notebook
    """
    def init_validation_tqdm(self):
        bar = tqdm(disable=True,)
        return bar