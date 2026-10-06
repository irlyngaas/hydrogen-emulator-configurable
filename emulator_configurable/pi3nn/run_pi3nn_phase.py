"""
CLI entry point for PI3NN's chained-SLURM-jobs orchestration: ONE phase
per invocation (mean, up, down, or calibrate). Each training phase
writes its resulting checkpoint path to a small text file so the next
chained SLURM job (--dependency=afterok) can read it without needing
complex inter-job argument passing. net_mean trains once and is reused
across however many up_down_mode sweeps you run (pass --mean-ckpt to
skip retraining it) -- it's identical regardless of up_down_mode, so
retraining it per sweep would be pure waste.

This is called once per phase rather than one script calling
train_model()/trainer.fit() three times in-process: constructing a
fresh pl.Trainer(strategy='ddp', ...) multiple times in one running
process is an untested combination here and a known source of friction
in some Lightning versions -- not worth risking on a real multi-GPU
run without testing it first. See run_pi3nn_phase.slurm +
submit_pi3nn_chain.sh for how the phases get chained on Frontier.

Usage:
  python -m emulator_configurable.pi3nn.run_pi3nn_phase --config <path> --phase mean
  python -m emulator_configurable.pi3nn.run_pi3nn_phase --config <path> --phase up --up-down-mode stateless_hidden
  python -m emulator_configurable.pi3nn.run_pi3nn_phase --config <path> --phase down --up-down-mode stateless_hidden
  python -m emulator_configurable.pi3nn.run_pi3nn_phase --config <path> --phase calibrate --up-down-mode stateless_hidden
"""
import argparse
import json

from ..train import train_model
from .calibration import calibrate


def _ckpt_file(config, role):
    # experiment_name, not run_name: run_name is the real CONUS1 .pfb data
    # directory name (must stay constant across every phase) --
    # experiment_name is just this set of PI3NN runs' human-readable label,
    # used only for naming logs/checkpoints/output files.
    return f"{config['logging_location']}/{config['experiment_name']}_{role}_ckpt.txt"


def _write_ckpt(config, role, ckpt_path):
    with open(_ckpt_file(config, role), 'w') as f:
        f.write(ckpt_path)


def _read_ckpt(config, role):
    with open(_ckpt_file(config, role)) as f:
        return f.read().strip()


def _checkpoint_monitor(config):
    return 'val_loss' if config['valid_fraction'] > 0 else 'train_loss'


def run_mean(config):
    ckpt = train_model(
        run_name=f"{config['experiment_name']}_mean",
        data_run_name=config['run_name'],  # the real .pfb data directory name -- see train.py's data_run_name docstring
        model_type='ForcedSTRNN',
        model_config=config['mean_model_config'],
        data_dir=config['data_dir'], parameter_list=config['parameter_list'],
        param_nlayer=config['param_nlayer'], patch_size=config['patch_size'],
        overlap=config['overlap'], max_epochs=config['max_epochs']['mean'],
        learning_rate=config['learning_rate']['mean'], sequence_length=config['sequence_length'],
        n_evaptrans=config['n_evaptrans'], batch_size=config['batch_size'],
        num_workers=config['num_workers'], precision=config['precision'],
        gradient_loss_penalty=config['gradient_loss_penalty'], device=config['device'],
        logging_location=config['logging_location'], scaler_file=config.get('scaler_file'),
        valid_fraction=config['valid_fraction'],
        early_stopping_patience=config['early_stopping_patience'],
        checkpoint_monitor=_checkpoint_monitor(config),
    )
    _write_ckpt(config, 'mean', ckpt)
    print(f'[mean] checkpoint: {ckpt}')
    return ckpt


def _check_hidden_state_channel(config, up_down_mode):
    """stateless_hidden's updown_config.hidden_state_channel must equal
    net_mean's own num_hidden[-1] (the channel count of the hidden state
    it feeds in as extra input) -- there's no code-level link between
    the two config blocks, so assert it here rather than let it fail as
    a silent shape mismatch deep inside StatelessUpDownNet."""
    if up_down_mode != 'stateless_hidden':
        return
    expected = config['mean_model_config']['num_hidden'][-1]
    actual = config['updown_model_config']['stateless_hidden'].get('hidden_state_channel')
    if actual != expected:
        raise ValueError(
            f"updown_model_config.stateless_hidden.hidden_state_channel ({actual}) must equal "
            f"mean_model_config.num_hidden[-1] ({expected})."
        )


def run_updown(config, role, up_down_mode, mean_ckpt_path=None):
    _check_hidden_state_channel(config, up_down_mode)
    mean_ckpt = mean_ckpt_path or _read_ckpt(config, 'mean')
    ckpt = train_model(
        run_name=f"{config['experiment_name']}_{up_down_mode}_{role}",
        data_run_name=config['run_name'],  # the real .pfb data directory name -- see train.py's data_run_name docstring
        model_type='PI3NNUpDownModule',
        model_config={
            'up_down_mode': up_down_mode, 'role': role,
            'mean_model_type': 'ForcedSTRNN', 'mean_model_config': config['mean_model_config'],
            'mean_ckpt_path': mean_ckpt,
            'updown_config': config['updown_model_config'][up_down_mode],
        },
        data_dir=config['data_dir'], parameter_list=config['parameter_list'],
        param_nlayer=config['param_nlayer'], patch_size=config['patch_size'],
        overlap=config['overlap'], max_epochs=config['max_epochs'][role],
        learning_rate=config['learning_rate'][role], sequence_length=config['sequence_length'],
        n_evaptrans=config['n_evaptrans'], batch_size=config['batch_size'],
        num_workers=config['num_workers'], precision=config['precision'],
        gradient_loss_penalty=False,  # PI3NNUpDownModule.configure_loss is a no-op; this flag is unused
        device=config['device'],
        logging_location=config['logging_location'], scaler_file=config.get('scaler_file'),
        valid_fraction=config['valid_fraction'],
        early_stopping_patience=config['early_stopping_patience'],
        checkpoint_monitor=_checkpoint_monitor(config),
    )
    _write_ckpt(config, f'{up_down_mode}_{role}', ckpt)
    print(f'[{up_down_mode}/{role}] checkpoint: {ckpt}')
    return ckpt


def run_calibrate(config, up_down_mode):
    mean_ckpt = _read_ckpt(config, 'mean')
    up_ckpt = _read_ckpt(config, f'{up_down_mode}_up')
    down_ckpt = _read_ckpt(config, f'{up_down_mode}_down')
    return calibrate(config, mean_ckpt, up_ckpt, down_ckpt, up_down_mode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--phase', required=True, choices=['mean', 'up', 'down', 'calibrate'])
    parser.add_argument('--up-down-mode', choices=['stateless_output', 'stateless_hidden', 'recurrent_clone'])
    parser.add_argument(
        '--mean-ckpt', default=None,
        help='Reuse an already-trained mean checkpoint instead of reading the file the mean phase writes -- '
             'lets you run multiple up_down_mode sweeps without retraining net_mean each time.',
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    if args.phase == 'mean':
        run_mean(config)
    elif args.phase in ('up', 'down'):
        if not args.up_down_mode:
            parser.error('--up-down-mode is required for --phase up/down')
        run_updown(config, args.phase, args.up_down_mode, mean_ckpt_path=args.mean_ckpt)
    elif args.phase == 'calibrate':
        if not args.up_down_mode:
            parser.error('--up-down-mode is required for --phase calibrate')
        run_calibrate(config, args.up_down_mode)


if __name__ == '__main__':
    main()
