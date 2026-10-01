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
    #
    # Every rank constructs its own logger instance here (Lightning expects
    # that), but log_hyperparams()/shutil.copy() below are plain, unguarded
    # Python code running *before* pl.Trainer even exists -- Lightning's
    # rank-aware logger-writing guarantees only apply once trainer.fit()
    # actually runs. Without a manual rank-0 guard, every SLURM-launched
    # process computes CSVLogger's "next version number" from an independent,
    # uncoordinated filesystem scan at the same moment and can collide on the
    # same directory (confirmed: an 8-task run raced on the same version_N
    # and crashed with "IsADirectoryError" at shutil.copy). Same principle
    # as the is_main_process guard emulator-1ts's main.py needed.
    rank = int(os.environ.get('SLURM_PROCID', 0))
    logger = pl.loggers.CSVLogger(
        save_dir=logging_location,
        name=run_name,
    )
    if rank == 0:
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
    # string like 'cuda:0'.
    #
    # devices= must equal SLURM_NTASKS_PER_NODE, not 1 -- this went through
    # two wrong guesses before landing here, both confirmed by actually
    # submitting jobs:
    #   devices=<ntasks-per-node> with --gpus-per-task=1 --gpu-bind=closest:
    #     CUDAAccelerator.parse_devices() fails ("You requested gpu:
    #     [0..7] But your machine only has: [0]"), since gpu-bind restricts
    #     each process to exactly 1 visible GPU.
    #   devices=1 (still with --gpus-per-task=1 --gpu-bind=closest):
    #     SLURMEnvironment.validate_settings() fails instead ("devices=1 ...
    #     does not match ... HINT: Set devices=8").
    # These two Lightning-internal checks are mutually exclusive under
    # per-task GPU-visibility restriction -- a known Lightning/SLURM
    # compatibility conflict (Lightning-AI/pytorch-lightning#16828), not
    # something specific to this setup. The actual fix is in the SLURM
    # script: request GPUs at the node level (--gpus-per-node, no
    # --gpus-per-task/--gpu-bind) so all of them stay visible to every
    # process, and devices= really does equal SLURM_NTASKS_PER_NODE --
    # Lightning's own SLURM-aware strategy then picks the correct one per
    # process internally via SLURM_LOCALID, unlike emulator-1ts's
    # hand-rolled main.py, which needed --gpus-per-task=1 specifically so
    # its own local_rank % device_count() correction had something to do.
    if str(device).startswith('cuda'):
        accelerator = 'gpu'
        devices = int(os.environ.get('SLURM_NTASKS_PER_NODE', 1))
        num_nodes = int(os.environ.get('SLURM_NNODES', 1))
    else:
        accelerator = 'cpu'
        devices = 1
        num_nodes = 1
    world_size = int(os.environ.get('SLURM_NTASKS', 1))
    strategy = 'ddp' if (accelerator == 'gpu' and world_size > 1) else 'auto'
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