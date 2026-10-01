import os
import shutil
import torch
import pytorch_lightning as pl

from typing import List, Union, Optional
from torch.utils.data import DataLoader
from .model_builder import model_setup
from .pfb_dataset import ParFlowSequenceDataset
from .scalers import create_scalers_from_yaml
from pytorch_lightning.callbacks import (
    Callback,
    ModelCheckpoint,
    LearningRateMonitor
)
from .utils import MetricsCallback

def train_model(
    run_name: str,
    model_type: str,
    model_config: dict,
    data_dir: str,
    parameter_list: List[str],
    param_nlayer: List[int],
    patch_size: int,
    overlap: int,
    max_epochs: int,
    learning_rate: float,
    sequence_length: int,
    *,
    n_evaptrans: int=0,
    batch_size: int=1,
    num_workers: int=1,
    precision: str='16',
    resume_from_checkpoint: Optional[str]=None,
    gradient_loss_penalty: bool=True,
    logging_frequency: int=10,
    callbacks: List[Callback]=[],
    device: Union[torch.device, str]='cuda',
    logging_location: str='./logs',
    scaler_file: Optional[str]=None,
    config_file: Optional[str]=None
):
    # Set up callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')
    metrics = MetricsCallback()
    checkpoint = ModelCheckpoint(
        save_top_k=5,
        every_n_train_steps=logging_frequency,
        every_n_epochs=None,
        monitor='train_loss'
    )
    callbacks = [lr_monitor, metrics, checkpoint]

    # resume_from_checkpoint is now just a local .ckpt path (or None) --
    # the MLflow-database-lookup option is gone along with the MLflow logger
    # below, since this path has no tracking server to query.
    ckpt_path = resume_from_checkpoint

    # Local, file-based logging -- no tracking server/credentials required.
    # logging_location is a local directory (PyTorch Lightning's CSVLogger
    # writes metrics.csv + hparams.yaml under
    # <logging_location>/<run_name>/version_N/).
    logger = pl.loggers.CSVLogger(
        save_dir=logging_location,
        name=run_name,
    )
    logger.log_hyperparams({
        k: v for k, v in locals().items()
        if k not in ('logger', 'callbacks') and isinstance(v, (int, float, str, bool))
    })
    if config_file:
        # CSVLogger has no artifact store like MLflow's -- just copy the
        # config alongside the run's own log directory instead.
        shutil.copy(config_file, logger.log_dir)

    # Set up the model. device=None disables model_setup's own internal
    # model.to(device) call (it otherwise defaults to a hardcoded 'cuda'
    # regardless of what's passed in here) -- under DDP, Lightning has to
    # own device placement itself; manually pre-moving the model to one
    # hardcoded device before trainer.fit() would put every process's model
    # on the same GPU instead of letting each process get its own.
    model = model_setup(
        model_type=model_type,
        model_config=model_config,
        learning_rate=learning_rate,
        gradient_loss_penalty=gradient_loss_penalty,
        device=None,
    )

    # Create the data loading pipeline. ForcedSTRNN has no built-in
    # scale_pressure/scale_statics/scale_evaptrans the way emulator-1ts's
    # ResNet does, so scaling happens inside the dataset itself here instead
    # of upstream in a data pipe.
    scalers = create_scalers_from_yaml(scaler_file) if scaler_file else None
    dataset = ParFlowSequenceDataset(
        data_dir=data_dir,
        run_name=run_name,
        parameter_list=parameter_list,
        patch_size=patch_size,
        overlap=overlap,
        param_nlayer=param_nlayer,
        sequence_length=sequence_length,
        n_evaptrans=n_evaptrans,
        scalers=scalers,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    # Configure the trainer.
    # Lightning's accelerator wants 'cpu'/'gpu'/'auto', not a raw device
    # string like 'cuda:0'. devices=/num_nodes= describe GPUs-per-node and
    # node count; derived from SLURM's own env vars (set by sbatch/srun, so
    # these fall back to 1/1 -- today's single-process behavior, unchanged --
    # whenever this isn't running under SLURM at all, exactly like the plain
    # interactive run that already worked).
    #
    # Launched via srun with one task per GPU (--gpus-per-task=1
    # --gpu-bind=closest, same pattern as emulator-1ts's
    # run_conus1_training.slurm), Lightning's SLURM-aware DDP strategy
    # handles rank/local-rank/process-group setup itself, including mapping
    # each process's one visible GPU correctly -- unlike emulator-1ts's
    # hand-rolled main.py, none of the local_rank % device_count() correction
    # work from that effort needs repeating here; Lightning's SLURM
    # integration already does the equivalent internally.
    if str(device).startswith('cuda'):
        accelerator = 'gpu'
        devices = int(os.environ.get('SLURM_NTASKS_PER_NODE', 1))
        num_nodes = int(os.environ.get('SLURM_NNODES', 1))
    else:
        accelerator = 'cpu'
        devices = 1
        num_nodes = 1
    strategy = 'ddp' if (accelerator == 'gpu' and devices * num_nodes > 1) else 'auto'
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        num_nodes=num_nodes,
        strategy=strategy,
        callbacks=callbacks,
        precision=precision,
        max_epochs=max_epochs,
        num_sanity_val_steps=0,
        log_every_n_steps=logging_frequency,
        logger=logger,
        gradient_clip_val=1.5,
        gradient_clip_algorithm="norm"
    )

    # Train the model
    trainer.fit(
        model=model,
        train_dataloaders=data_loader,
        ckpt_path=ckpt_path
    )

    logger.finalize()