"""
Small-scale validation for the emulator_configurable PI3NN integration,
mirroring emulator-1ts/pi3nn/validate_synthetic.py's role: a standalone,
runnable correctness check (this repo's test/ package is an empty
placeholder, no pytest infrastructure exists anywhere -- matching
existing convention, not introducing one here) before trusting any of
this on real Frontier-scale CONUS1 data.

Everything emulator-1ts's validate_synthetic.py already covers (the
positive-output requirement, the DDP masked-loss gradient match via a
real DDP-wrapped model + deliberately uneven mask counts, PICP-matches-
quantile) transfers directly here since losses.py/boundary_optimizer.py
are vendored unchanged -- this file focuses on what's genuinely new:
the three-way architecture switch, the return_hidden additive change to
ForcedSTRNN, StatelessUpDownNet's vectorization, freeze correctness
under Lightning's automatic_optimization, recurrent_clone's per-
timestep positivity, and the full mean->up->down->calibrate sequence on
synthetic data via train_model()'s dataset-injection seam.

Usage: python -m emulator_configurable.pi3nn.validate_synthetic
       torchrun --nproc_per_node=2 --standalone -m emulator_configurable.pi3nn.validate_synthetic --mode ddp_loss
"""
import argparse
import os
import tempfile

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset

from ..models import ForcedSTRNN
from ..train import train_model
from .calibration import calibrate
from .lightning_modules import PI3NNUpDownModule
from .losses import reduced_masked_mse_loss
from .networks import build_updown_net, predict_updown

UP_DOWN_MODES = ['stateless_output', 'stateless_hidden', 'recurrent_clone']


class SyntheticSequenceDataset(Dataset):
    """Shapes match ParFlowSequenceDataset.__getitem__ exactly. img_channel
    == init_cond_channel == out_channel is required for ForcedSTRNN's
    residual update (x = conv_last(h) + x) to be shape-consistent across
    timesteps -- confirmed from the real boxtest config, where all three
    are set to the same value (5)."""

    def __init__(self, n, sequence_length, act_channel, static_channel, state_channel, patch, seed):
        g = torch.Generator().manual_seed(seed)
        self.forcings = torch.randn(n, sequence_length, act_channel, patch, patch, generator=g)
        self.init_cond = torch.randn(n, 1, state_channel, patch, patch, generator=g)
        self.static_inputs = torch.randn(n, 1, static_channel, patch, patch, generator=g)
        self.target = torch.randn(n, sequence_length, state_channel, patch, patch, generator=g)

    def __len__(self):
        return self.forcings.shape[0]

    def __getitem__(self, idx):
        return self.forcings[idx], self.init_cond[idx], self.static_inputs[idx], self.target[idx]


def make_mean_config():
    return dict(num_layers=1, num_hidden=[8], img_channel=3, act_channel=2, init_cond_channel=3, static_channel=4, out_channel=3)


def make_updown_config(up_down_mode, mean_config):
    if up_down_mode == 'stateless_output':
        return dict(act_channel=2, static_channel=4, out_channel=3, hidden_dim=8, kernel_size=3, depth=1)
    elif up_down_mode == 'stateless_hidden':
        return dict(
            act_channel=2, static_channel=4, out_channel=3,
            hidden_state_channel=mean_config['num_hidden'][-1],
            hidden_dim=8, kernel_size=3, depth=1,
        )
    elif up_down_mode == 'recurrent_clone':
        return dict(
            num_layers=1, num_hidden=[8], act_channel=2,
            init_cond_channel=3, static_channel=4, out_channel=3,
        )
    raise ValueError(up_down_mode)


def check_positivity_all_modes():
    torch.manual_seed(0)
    mean_config = make_mean_config()
    net_mean = ForcedSTRNN(**mean_config).eval()
    for p in net_mean.parameters():
        p.requires_grad_(False)

    forcing = torch.randn(2, 4, mean_config['act_channel'], 8, 8)
    state = torch.randn(2, 1, mean_config['init_cond_channel'], 8, 8)
    params = torch.randn(2, 1, mean_config['static_channel'], 8, 8)

    for mode in UP_DOWN_MODES:
        net = build_updown_net(mode, make_updown_config(mode, mean_config))
        pred, _ = predict_updown(net_mean, net, mode, forcing, state, params)
        assert (pred > 0).all(), f'{mode}: output not strictly positive'
    print('PASS: strictly-positive output for all three up_down_mode variants.')


def check_return_hidden_preserves_behavior():
    torch.manual_seed(1)
    mean_config = make_mean_config()
    net_mean = ForcedSTRNN(**mean_config).eval()

    forcing = torch.randn(2, 4, mean_config['act_channel'], 8, 8)
    state = torch.randn(2, 1, mean_config['init_cond_channel'], 8, 8)
    params = torch.randn(2, 1, mean_config['static_channel'], 8, 8)

    with torch.no_grad():
        out_plain = net_mean(forcing, state, params)
        out_pair = net_mean(forcing, state, params, return_hidden=True)

    assert isinstance(out_pair, tuple) and len(out_pair) == 2
    assert torch.equal(out_plain, out_pair[0]), 'return_hidden=False call site is not behavior-preserving'
    expected_hidden_shape = (2, 4, mean_config['num_hidden'][-1], 8, 8)
    assert out_pair[1].shape == expected_hidden_shape, f'hidden shape {out_pair[1].shape} != {expected_hidden_shape}'
    print('PASS: return_hidden=False is bit-identical to the original call site; hidden tensor shape correct.')


def check_stateless_vectorization():
    torch.manual_seed(2)
    mean_config = make_mean_config()
    net = build_updown_net('stateless_output', make_updown_config('stateless_output', mean_config)).eval()

    batch, T, H, W = 2, 4, 8, 8
    forcings = torch.randn(batch, T, mean_config['act_channel'], H, W)
    static_inputs = torch.randn(batch, 1, mean_config['static_channel'], H, W)
    mean_out = torch.randn(batch, T, mean_config['out_channel'], H, W)

    with torch.no_grad():
        vectorized = net(forcings, static_inputs, mean_out)

        static = static_inputs[:, 0]
        ref_frames = []
        for t in range(T):
            x = torch.cat([forcings[:, t], static, mean_out[:, t]], dim=1)
            ref_frames.append(net.positive(net.backbone(x)))
        reference = torch.stack(ref_frames, dim=1)

    assert torch.allclose(vectorized, reference, atol=1e-6), 'vectorized StatelessUpDownNet path disagrees with per-timestep reference loop'
    print('PASS: StatelessUpDownNet vectorization matches a per-timestep reference loop.')


def check_freeze_correctness():
    torch.manual_seed(3)
    mean_config = make_mean_config()
    net_mean = ForcedSTRNN(**mean_config)
    with tempfile.TemporaryDirectory() as d:
        ckpt_path = os.path.join(d, 'mean.ckpt')
        torch.save({'state_dict': net_mean.state_dict()}, ckpt_path)

        updown_mode = 'stateless_output'
        module = PI3NNUpDownModule(
            up_down_mode=updown_mode, role='up',
            mean_model_type='ForcedSTRNN', mean_model_config=mean_config, mean_ckpt_path=ckpt_path,
            updown_config=make_updown_config(updown_mode, mean_config),
        )
        module.learning_rate = 0.01
        module.train()

        before = {k: v.clone() for k, v in module.net_mean.state_dict().items()}

        batch = (
            torch.randn(2, 4, mean_config['act_channel'], 8, 8),
            torch.randn(2, 1, mean_config['init_cond_channel'], 8, 8),
            torch.randn(2, 1, mean_config['static_channel'], 8, 8),
            torch.randn(2, 4, mean_config['out_channel'], 8, 8),
        )
        optimizer = module.configure_optimizers()
        for _ in range(3):
            optimizer.zero_grad()
            loss = module.training_step(batch, 0)
            loss.backward()
            optimizer.step()

        after = module.net_mean.state_dict()
        for k in before:
            assert torch.equal(before[k], after[k]), f'net_mean.{k} changed after training_step -- freeze is broken'
        assert module.net_mean.training is False, 'net_mean left in train mode after module.train()'
    print('PASS: net_mean is bit-identical and stays in eval mode after training_steps (freeze correctness).')


def check_recurrent_clone_positivity():
    torch.manual_seed(4)
    mean_config = make_mean_config()
    net = build_updown_net('recurrent_clone', make_updown_config('recurrent_clone', mean_config))

    forcing = torch.randn(2, 5, mean_config['act_channel'], 8, 8)
    state = torch.randn(2, 1, mean_config['init_cond_channel'], 8, 8)
    params = torch.randn(2, 1, mean_config['static_channel'], 8, 8)
    with torch.no_grad():
        out = net(forcing, state, params)
    # out already contains every timestep's fed-forward x stacked (dim=1)
    # -- positivity of the whole tensor IS positivity at every timestep.
    assert (out > 0).all(), 'recurrent_clone is not strictly positive at every timestep'
    print('PASS: recurrent_clone output is strictly positive at every timestep.')


def check_full_sequence():
    torch.manual_seed(5)
    mean_config = make_mean_config()
    seq_len = 3
    train_ds = SyntheticSequenceDataset(16, seq_len, mean_config['act_channel'], mean_config['static_channel'], mean_config['out_channel'], 8, seed=10)
    valid_ds = SyntheticSequenceDataset(8, seq_len, mean_config['act_channel'], mean_config['static_channel'], mean_config['out_channel'], 8, seed=11)

    common = dict(
        data_dir='', parameter_list=[], param_nlayer=[], patch_size=0, overlap=0,
        sequence_length=seq_len, batch_size=4, num_workers=0, precision='32',
        device='cpu', gradient_loss_penalty=False, logging_frequency=1,
        valid_fraction=0.0, early_stopping_patience=None,
    )

    with tempfile.TemporaryDirectory() as logdir:
        mean_ckpt = train_model(
            run_name='synthetic_mean', model_type='ForcedSTRNN', model_config=mean_config,
            max_epochs=2, learning_rate=0.01, logging_location=logdir,
            train_dataset=train_ds, valid_dataset=valid_ds, **common,
        )
        assert mean_ckpt, 'mean phase produced no checkpoint path -- likely every_n_train_steps too high for this tiny dataset'

        for up_down_mode in UP_DOWN_MODES:
            updown_config = make_updown_config(up_down_mode, mean_config)
            role_ckpts = {}
            for role in ('up', 'down'):
                role_ckpts[role] = train_model(
                    run_name=f'synthetic_{up_down_mode}_{role}', model_type='PI3NNUpDownModule',
                    model_config={
                        'up_down_mode': up_down_mode, 'role': role,
                        'mean_model_type': 'ForcedSTRNN', 'mean_model_config': mean_config,
                        'mean_ckpt_path': mean_ckpt, 'updown_config': updown_config,
                    },
                    max_epochs=2, learning_rate=0.01, logging_location=logdir,
                    train_dataset=train_ds, valid_dataset=valid_ds, **common,
                )
                assert role_ckpts[role], f'{up_down_mode}/{role} produced no checkpoint path'

            results = calibrate(
                config={
                    'mean_model_config': mean_config,
                    'updown_model_config': {up_down_mode: updown_config},
                    'quantile': 0.9, 'logging_location': logdir, 'experiment_name': f'synthetic_{up_down_mode}',
                    'batch_size': common['batch_size'], 'num_workers': common['num_workers'],
                },
                mean_ckpt_path=mean_ckpt, up_ckpt_path=role_ckpts['up'], down_ckpt_path=role_ckpts['down'],
                up_down_mode=up_down_mode, train_dataset=train_ds,
            )
            assert len(results) == mean_config['out_channel']
            print(f'[{up_down_mode}] calibration results: {results}')

    print('PASS: full mean -> up -> down -> calibrate sequence completes for all three up_down_mode variants.')


def run_full():
    check_positivity_all_modes()
    check_return_hidden_preserves_behavior()
    check_stateless_vectorization()
    check_freeze_correctness()
    check_recurrent_clone_positivity()
    check_full_sequence()
    print('PASS: all synthetic validation checks passed.')


def run_ddp_loss_check():
    """Identical in spirit to emulator-1ts/pi3nn/validate_synthetic.py's
    ddp_loss check -- losses.py is vendored unchanged, so this is the
    same check, just kept here too so this package's validation is
    self-contained."""
    if 'WORLD_SIZE' in os.environ:
        rank, world_size = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    elif 'SLURM_NTASKS' in os.environ:
        rank, world_size = int(os.environ['SLURM_PROCID']), int(os.environ['SLURM_NTASKS'])
    else:
        rank, world_size = 0, 1
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
    assert world_size == 2, 'run_ddp_loss_check expects exactly 2 processes'

    torch.manual_seed(0)
    full_x = torch.randn(40, dtype=torch.float64)
    full_target = torch.randn(40, dtype=torch.float64)
    full_mask = torch.zeros(40, dtype=torch.bool)
    full_mask[torch.randperm(20)[:18]] = True
    full_mask[20 + torch.randperm(20)[:3]] = True

    class TinyLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float64))
            self.b = torch.nn.Parameter(torch.tensor(0.1, dtype=torch.float64))

        def forward(self, x):
            return self.w * x + self.b

    model = TinyLinear()
    ddp_model = DDP(model)

    local_slice = slice(0, 20) if rank == 0 else slice(20, 40)
    pred_local = ddp_model(full_x[local_slice])
    loss = reduced_masked_mse_loss(pred_local, full_target[local_slice], full_mask[local_slice])
    loss.backward()
    w_grad, b_grad = model.w.grad.item(), model.b.grad.item()

    ref_model = TinyLinear()
    with torch.no_grad():
        ref_model.w.copy_(model.w)
        ref_model.b.copy_(model.b)
    ref_pred = ref_model(full_x)
    ref_loss = ((ref_pred - full_target) ** 2 * full_mask.to(ref_pred.dtype)).sum() / full_mask.sum().clamp_min(1).to(ref_pred.dtype)
    ref_loss.backward()
    ref_w_grad, ref_b_grad = ref_model.w.grad.item(), ref_model.b.grad.item()

    w_diff, b_diff = abs(w_grad - ref_w_grad), abs(b_grad - ref_b_grad)
    print(f'[rank {rank}] w_grad diff={w_diff:.2e} b_grad diff={b_diff:.2e}')
    assert w_diff < 1e-8 and b_diff < 1e-8
    dist.barrier()
    if rank == 0:
        print("PASS: reduced_masked_mse_loss's DDP-averaged gradient matches the non-distributed reference.")
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['full', 'ddp_loss'], default='full')
    args = parser.parse_args()
    if args.mode == 'full':
        run_full()
    else:
        run_ddp_loss_check()
