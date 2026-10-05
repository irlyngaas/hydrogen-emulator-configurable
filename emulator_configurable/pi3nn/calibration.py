"""
Post-training calibration + evaluation, adapted from
emulator-1ts/pi3nn/trainer.py's boundary_optimization()/evaluate(). Run
once per up_down_mode, after all three trainer.fit() calls (mean, up,
down) for that mode have completed -- not Lightning-managed itself,
just a plain single-process script (this is the "calibrate" phase of
run_pi3nn_phase.py's chained-SLURM-jobs design, so there's no DDP/rank
concept here at all, unlike emulator-1ts's rank-0-only pattern).

Per-channel (not pooled) bisection calibration via the vendored,
unmodified BoundaryOptimizer -- same requirement as emulator-1ts:
pooling all channels together would calibrate to the aggregate
coverage rather than each channel's own, defeating the point of
per-channel calibration. Here "per-channel" flattens across
(batch, timestep, H, W) jointly for each channel, since the extra
timestep axis is just another dimension to flatten away, not a reason
to calibrate separately per-timestep.
"""
import json

import torch
from torch.utils.data import DataLoader

from .. import model_builder
from ..pfb_dataset import ParFlowSequenceDataset
from ..scalers import create_scalers_from_yaml
from .boundary_optimizer import BoundaryOptimizer
from .networks import build_updown_net, predict_updown


def _load_mean(mean_model_type, mean_model_config, mean_ckpt_path):
    model = model_builder.ModelBuilder.build_emulator(mean_model_type, dict(mean_model_config))
    ckpt = torch.load(mean_ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _load_updown(up_down_mode, updown_config, ckpt_path):
    net = build_updown_net(up_down_mode, updown_config)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    # PI3NNUpDownModule's own checkpoint contains BOTH net_mean.* (its own
    # frozen copy, reloaded separately above anyway) and net.* (the actual
    # trained up/down network) -- extract just the latter and strip the
    # prefix before loading into the bare net built here.
    state = {k[len('net.'):]: v for k, v in ckpt['state_dict'].items() if k.startswith('net.')}
    net.load_state_dict(state)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def _caps(y, mean, up, down, c_up, c_down):
    """y/mean/up/down: 1D numpy arrays, one channel's worth, flattened
    across every sample/timestep/pixel. c_up/c_down: scalars for this
    channel."""
    import numpy as np
    upper = mean + c_up * up
    lower = mean - c_down * down
    inside = (y <= upper) & (y >= lower)
    picp = float(np.mean(inside))
    mpiw = float(np.mean(upper - lower))
    mse = float(np.mean((mean - y) ** 2))
    rmse = mse ** 0.5
    ss_res = np.sum((y - mean) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = float(1 - ss_res / max(ss_tot, 1e-12))
    return {'picp': picp, 'mpiw': mpiw, 'rmse': rmse, 'r2': r2}


def calibrate(config, mean_ckpt_path, up_ckpt_path, down_ckpt_path, up_down_mode, train_dataset=None):
    """train_dataset: bypasses the ParFlowSequenceDataset(data_dir=...)
    construction below when supplied -- same seam train_model() has,
    needed for validate_synthetic.py to run this on in-memory tensors
    instead of real .pfb files."""
    net_mean = _load_mean('ForcedSTRNN', config['mean_model_config'], mean_ckpt_path)
    net_up = _load_updown(up_down_mode, config['updown_model_config'][up_down_mode], up_ckpt_path)
    net_down = _load_updown(up_down_mode, config['updown_model_config'][up_down_mode], down_ckpt_path)

    if train_dataset is not None:
        train_ds = train_dataset
    else:
        scalers = create_scalers_from_yaml(config['scaler_file']) if config.get('scaler_file') else None
        train_ds = ParFlowSequenceDataset(
            data_dir=config['data_dir'], run_name=config['run_name'],
            parameter_list=config['parameter_list'], patch_size=config['patch_size'],
            overlap=config['overlap'], param_nlayer=config['param_nlayer'],
            sequence_length=config['sequence_length'], n_evaptrans=config['n_evaptrans'],
            scalers=scalers, valid_fraction=config['valid_fraction'], split='train',
        )
    train_dl = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    means, ups, downs, ys = [], [], [], []
    with torch.no_grad():
        for batch in train_dl:
            forcing, state, params, target = batch
            pred_up, mean_pred = predict_updown(net_mean, net_up, up_down_mode, forcing, state, params)
            pred_down, _ = predict_updown(net_mean, net_down, up_down_mode, forcing, state, params)
            means.append(mean_pred)
            ups.append(pred_up)
            downs.append(pred_down)
            ys.append(target)
    mean_t = torch.cat(means)
    up_t = torch.cat(ups)
    down_t = torch.cat(downs)
    y_t = torch.cat(ys)

    out_channel = mean_t.shape[2]
    quantile = config['quantile']
    results = {}
    for c in range(out_channel):
        y_c = y_t[:, :, c].flatten().numpy()
        m_c = mean_t[:, :, c].flatten().numpy()
        u_c = up_t[:, :, c].flatten().numpy()
        d_c = down_t[:, :, c].flatten().numpy()
        num_outlier = int(y_c.shape[0] * (1 - quantile) / 2)
        opt = BoundaryOptimizer(
            y_c, m_c, u_c, d_c, num_outlier=num_outlier,
            c_up0_ini=0.0, c_up1_ini=100000.0,
            c_down0_ini=0.0, c_down1_ini=100000.0, max_iter=1000,
        )
        c_up = opt.optimize_up(verbose=0)
        c_down = opt.optimize_down(verbose=0)
        results[f'channel_{c}'] = {'c_up': c_up, 'c_down': c_down, **_caps(y_c, m_c, u_c, d_c, c_up, c_down)}

    out_path = f"{config['logging_location']}/{config['run_name']}_{up_down_mode}_calibration.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved calibration results to {out_path}')
    for k, v in results.items():
        print(k, v)
    return results
