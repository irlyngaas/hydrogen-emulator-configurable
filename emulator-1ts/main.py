import os
import yaml
import torch
import torch.distributed as dist

from dataset import ParFlowDataset
from model import get_model
from train import train_model
from argparse import ArgumentParser
from utils import get_optimizer, get_loss, get_dtype
from scalers import create_scalers_from_yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def read_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_distributed_info():
    """
    Reads rank/world_size/local_rank from whichever launcher set them:
    torchrun-style env vars (RANK/WORLD_SIZE/LOCAL_RANK), or srun's
    (SLURM_PROCID/SLURM_NTASKS/SLURM_LOCALID). Defaults to a single,
    non-distributed process (0, 1, 0) if neither is present -- e.g. running
    `python main.py` directly, or `srun -n1 ...` with a single task.
    """
    if 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_NTASKS' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = int(os.environ['SLURM_LOCALID'])
    else:
        rank, world_size, local_rank = 0, 1, 0
    return rank, world_size, local_rank


def custom_collate(batch):
    s, e, p, y = [], [], [], []
    for b in batch:
        s.append(b[0])
        e.append(b[1])
        p.append(b[2])
        y.append(b[3])
    s = torch.stack(s)
    e = torch.stack(e)
    p = torch.stack(p)
    y = torch.stack(y)
    return s, e, p, y

def train(
    name: str,
    log_location: str,
    model_type: str,
    optimizer: str,
    loss: str,
    n_epochs: int,
    batch_size: int,
    lr: float,
    data_def: dict,
    model_def: dict,
    device: str,
    num_workers: int,
    dtype: str,
    **kwargs
):
    rank, world_size, local_rank = get_distributed_info()
    distributed = world_size > 1
    is_main_process = rank == 0

    if distributed:
        # With --gpus-per-task=1 --gpu-bind=closest (what run_conus1_training.slurm
        # uses), each process only ever sees ONE GPU, always at index 0 from
        # its own vantage point -- torch.cuda.set_device(local_rank) directly
        # fails for any rank > 0 with "invalid device ordinal" in that case.
        # Without per-task binding, every process sees all GPUs on the node
        # and needs local_rank to pick its own. The modulo handles both:
        # local_gpu_id is always 0 when only one GPU is visible, and the
        # correct distinct index when all are.
        local_gpu_id = local_rank % torch.cuda.device_count()
        # The device: value in the config is ignored here, since using it
        # verbatim for every process would put every rank on the same GPU
        device = f'cuda:{local_gpu_id}'
        torch.cuda.set_device(local_gpu_id)
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    # Create the data loader
    dtype = get_dtype(dtype)
    # scaler_yaml isn't a ParFlowDataset argument -- pull it out here and use
    # it to build the model's scalers dict below instead (previously this
    # key was either silently ignored or crashed the dataset constructor
    # with an unexpected-keyword-argument error, depending on whether it
    # was present in data_def).
    data_def = dict(data_def)
    scaler_yaml = data_def.pop('scaler_yaml', None)
    dataset = ParFlowDataset(**data_def, dtype=dtype)

    # DistributedSampler shards the dataset across ranks instead of every
    # process shuffling over the whole thing independently
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
    train_dl = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=custom_collate,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers
    )


    # Create the model
    # Add names of model inputs to model definition for scaling, if needed
    model_def['pressure_names'] = dataset.PRESSURE_NAMES
    model_def['evaptrans_names'] = dataset.EVAPTRANS_NAMES
    model_def['param_names'] = dataset.PARAM_NAMES
    model_def['n_evaptrans'] = dataset.n_evaptrans
    model_def['parameter_list'] = dataset.parameter_list
    model_def['param_nlayer'] = dataset.param_nlayer
    if scaler_yaml is not None:
        model_def['scalers'] = create_scalers_from_yaml(scaler_yaml)
    model = get_model(model_type, model_def)
    model = model.to(device).to(dtype)

    if distributed:
        # Must be local_gpu_id, not local_rank -- DDP uses device_ids directly
        # to pick the target device for moving forward-pass inputs (including
        # indexing its own internal per-visible-device stream cache), so the
        # same local_rank-vs-visible-device-count mismatch from set_device()
        # above applies here too.
        model = DDP(model, device_ids=[local_gpu_id])

    # Create the optimizer and loss function
    optimizer = get_optimizer(optimizer, model, lr)
    loss_fn = get_loss(loss)

    metrics = train_model(
        model, train_dl, optimizer, loss_fn, n_epochs, device=device,
        sampler=sampler,
    )

    # Only rank 0 logs/saves -- every process would otherwise redundantly
    # print the same metrics and race to write the same output files
    if is_main_process:
        print('----------------------------------------')
        print(metrics)
        print('----------------------------------------')

        metrics_filename = f'{log_location}/{name}_metrics.csv'
        weights_filename = f'{log_location}/{name}_weights_only.pth'
        model_filename = f'{log_location}/{name}_model.pth'
        metrics.to_csv(metrics_filename)
        # DDP only forwards forward()/__call__ -- the actual model (with its
        # custom scale_* methods and jit-exportable state) lives at .module
        raw_model = model.module if distributed else model
        torch.save(raw_model.state_dict(), weights_filename)
        m = torch.jit.script(raw_model)
        torch.jit.save(m, model_filename)

        print('----------------------------------------')
        print(f'Metrics saved to {metrics_filename}')
        print(f'Model saved to {model_filename}')

    if distributed:
        dist.destroy_process_group()


def test():
    pass

def main(config, mode):
    config = read_config(config)

    if mode == "train":
        print("TRAINING")
        train(**config)
    elif mode == "test":
        print("TESTING")
        # Note: Not implemented
        test(**config)


if __name__ == "__main__":
    # EXAMPLE USAGE: python main.py --config example_config.yaml --mode train
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--mode", type=str, required=True, choices=["train", "test"], default="train"
    )
    args = parser.parse_args()
    main(args.config, args.mode)
