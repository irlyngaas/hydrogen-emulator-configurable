"""
PI3NN training entry point for emulator-1ts, mirroring main.py's train()
as closely as possible -- same DDP/dataset/DataLoader setup, reusing
main.py's own get_distributed_info/custom_collate directly rather than
duplicating them (see pi3nn/distributed-setup note in the integration
plan: main.py is the ORIGINAL hand-rolled-DDP pattern this whole project
family's distributed training already follows, not something to
reinvent here).

EXAMPLE USAGE: python main_pi3nn.py --config pi3nn_example_config.yaml
"""
import os

import torch
import torch.distributed as dist
import yaml
from argparse import ArgumentParser
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: F401 (kept for parity with main.py's imports)
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset import ParFlowDataset
from main import get_distributed_info, custom_collate
from scalers import create_scalers_from_yaml
from utils import get_dtype
from pi3nn.networks import build_networks
from pi3nn.trainer import PI3NNConvTrainer, is_main_process


def read_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def custom_collate_with_coords(batch):
    """custom_collate handles exactly 4 elements per sample and would
    silently drop a 5th -- needed only for the coords-bearing loaders
    built when pi3nn_configs['calibration_mode'] == 'spatial_field' and
    pi3nn_configs['spatial_field_coords'] == 'absolute'."""
    s, e, p, y, c = [], [], [], [], []
    for b in batch:
        s.append(b[0])
        e.append(b[1])
        p.append(b[2])
        y.append(b[3])
        c.append(b[4])
    return torch.stack(s), torch.stack(e), torch.stack(p), torch.stack(y), torch.stack(c)


def train(
    name, log_location, data_def, model_def, pi3nn_configs,
    device, num_workers, dtype, batch_size, valid_fraction=0.1,
    **kwargs,
):
    rank, world_size, local_rank = get_distributed_info()
    distributed = world_size > 1

    if distributed:
        # Same local_gpu_id = local_rank % device_count() mapping as
        # main.py -- handles both --gpus-per-task=1 --gpu-bind=closest
        # (every process sees exactly 1 GPU, always index 0) and no
        # per-task binding (every process sees all GPUs, needs local_rank
        # to pick its own) with the same line.
        local_gpu_id = local_rank % torch.cuda.device_count()
        device = f'cuda:{local_gpu_id}'
        torch.cuda.set_device(local_gpu_id)
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    dtype = get_dtype(dtype)
    data_def = dict(data_def)
    scaler_yaml = data_def.pop('scaler_yaml', None)

    train_ds = ParFlowDataset(**data_def, dtype=dtype, valid_fraction=valid_fraction, split='train')
    valid_ds = ParFlowDataset(**data_def, dtype=dtype, valid_fraction=valid_fraction, split='valid')

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
    valid_sampler = DistributedSampler(valid_ds, num_replicas=world_size, rank=rank, shuffle=False) if distributed else None

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, collate_fn=custom_collate,
        shuffle=(train_sampler is None), sampler=train_sampler, num_workers=num_workers,
    )
    valid_dl = DataLoader(
        valid_ds, batch_size=batch_size, collate_fn=custom_collate,
        shuffle=False, sampler=valid_sampler, num_workers=num_workers,
    )
    # No sampler: a plain, unsharded pass over the complete training set,
    # only ever iterated by rank 0 (see PI3NNConvTrainer.boundary_optimization/
    # evaluate) -- the bisection search needs actual per-pixel values, not
    # something reducible to a sum/count across ranks the way MSE is.
    train_dl_full = DataLoader(
        train_ds, batch_size=batch_size, collate_fn=custom_collate,
        shuffle=False, num_workers=num_workers,
    )

    # Only built when pi3nn_configs asks for spatial-field calibration in
    # 'absolute' coordinate mode -- a SEPARATE pair of ParFlowDataset
    # instances (return_coords=True), not a return_coords flip on train_ds/
    # valid_ds themselves, so train_dl/valid_dl (used for ordinary SGD
    # training every epoch) keep their existing 4-tuple contract exactly as
    # it was. 'patch_relative' mode (the default) needs none of this --
    # it reuses train_dl_full/valid_dl directly, same as the scalar path.
    train_dl_full_coords = valid_dl_coords = None
    calibration_mode = pi3nn_configs.get('calibration_mode', 'scalar')
    spatial_field_coords = pi3nn_configs.get('spatial_field_coords', 'patch_relative')
    if calibration_mode == 'spatial_field' and spatial_field_coords == 'absolute':
        train_ds_coords = ParFlowDataset(**data_def, dtype=dtype, valid_fraction=valid_fraction, split='train', return_coords=True)
        valid_ds_coords = ParFlowDataset(**data_def, dtype=dtype, valid_fraction=valid_fraction, split='valid', return_coords=True)
        train_dl_full_coords = DataLoader(
            train_ds_coords, batch_size=batch_size, collate_fn=custom_collate_with_coords,
            shuffle=False, num_workers=num_workers,
        )
        valid_dl_coords = DataLoader(
            valid_ds_coords, batch_size=batch_size, collate_fn=custom_collate_with_coords,
            shuffle=False, num_workers=num_workers,
        )

    model_def = dict(model_def)
    model_def['pressure_names'] = train_ds.PRESSURE_NAMES
    model_def['evaptrans_names'] = train_ds.EVAPTRANS_NAMES
    model_def['param_names'] = train_ds.PARAM_NAMES
    model_def['n_evaptrans'] = train_ds.n_evaptrans
    model_def['parameter_list'] = train_ds.parameter_list
    model_def['param_nlayer'] = train_ds.param_nlayer
    if scaler_yaml is not None:
        model_def['scalers'] = create_scalers_from_yaml(scaler_yaml)

    # bias_init/eps control PositiveResNetWrapper's positivity-enforcing
    # activation (net_up/net_down only -- net_mean is unaffected and can be
    # reused across different eps values without retraining). Previously
    # hardcoded at build_networks' defaults; exposed here since eps=0.2's
    # floor (sqrt(0.2)~=0.447) turned out to be far larger than real CONUS1
    # residual scales given how well net_mean fits, causing net_up/net_down
    # to saturate at that floor almost everywhere (confirmed directly via
    # plot_prediction's raw-output logging) -- exactly the "worth a sanity
    # check against real CONUS1 residual magnitudes" PositiveResNetWrapper's
    # own docstring flagged before any real data existed.
    bias_init = pi3nn_configs.get('bias_init', 3.0)
    eps = pi3nn_configs.get('eps', 0.2)
    # Direct confirmation that the config value actually reached here --
    # up_pred/down_pred's floor has been observed stuck at sqrt(0.2) even
    # with eps:0.00001 set in the yaml, so printing the value immediately
    # before it's passed into build_networks settles whether the config
    # read itself is the problem, rather than inferring it indirectly from
    # the trained network's floor behavior several steps downstream.
    if is_main_process():
        print(f'[main_pi3nn.train] bias_init={bias_init!r}, eps={eps!r} (from pi3nn_configs.get, floor=sqrt(eps)={eps**0.5:.4g})')
    net_mean, net_up, net_down = build_networks(model_def, bias_init=bias_init, eps=eps)
    net_mean = net_mean.to(device).to(dtype)
    net_up = net_up.to(device).to(dtype)
    net_down = net_down.to(device).to(dtype)

    trainer = PI3NNConvTrainer(
        pi3nn_configs, net_mean, net_up, net_down,
        train_dl, valid_dl, train_dl_full,
        train_sampler=train_sampler, device=device,
        train_dl_full_coords=train_dl_full_coords, valid_dl_coords=valid_dl_coords,
    )
    trainer.train()

    verbose = pi3nn_configs.get('verbose', 0)
    save_dict = {
        'net_mean': trainer.net_mean.state_dict(),
        'net_up': trainer.net_up.state_dict(),
        'net_down': trainer.net_down.state_dict(),
        'model_def': model_def,
    }
    if calibration_mode == 'scalar':
        trainer.boundary_optimization(verbose=verbose)
        results = trainer.evaluate(verbose=verbose)
        save_dict['c_up'] = trainer.c_up
        save_dict['c_down'] = trainer.c_down
    elif calibration_mode == 'spatial_field':
        spatial_field_rank = pi3nn_configs.get('spatial_field_rank', 3)
        trainer.boundary_optimization_spatial_field(coords_mode=spatial_field_coords, rank=spatial_field_rank, verbose=verbose)
        results = trainer.evaluate_spatial_field(verbose=verbose)
        save_dict['c_up_field'] = trainer.c_up_field
        save_dict['c_down_field'] = trainer.c_down_field
        save_dict['spatial_field_coords'] = spatial_field_coords
    else:
        raise ValueError(f"unknown calibration_mode {calibration_mode!r}")
    save_dict['results'] = results

    if is_main_process():
        print('----------------------------------------')
        print(results)
        print('----------------------------------------')
        os.makedirs(log_location, exist_ok=True)
        torch.save(save_dict, f'{log_location}/{name}_pi3nn.pth')
        print(f'Saved to {log_location}/{name}_pi3nn.pth')

    if distributed:
        dist.destroy_process_group()


def main(config):
    config = read_config(config)
    train(**config)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    main(args.config)
