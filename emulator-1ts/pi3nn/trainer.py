"""
PI3NN trainer for emulator-1ts's 2D spatial data. Structurally mirrors
the validated flat PI3NN port (UQnet/pi3nn_torch/trainer.py): sequential
phases (train net_mean fully, freeze it, train net_up on its positive
residuals, train net_down on its negative residuals), per-phase early
stopping with a DDP-broadcast-synchronized decision so no rank's
stop/best-weights bookkeeping can diverge from rank 0's regardless of
any per-GPU floating-point non-determinism (identical mechanism to the
flat port's _train_one_network, just at epoch granularity instead of
per full-batch iteration, since real spatial data needs mini-batching).

The one piece of genuinely new mechanics here (not in the flat port) is
the masked loss for up/down -- see losses.py's module docstring for why
it needs its own DDP reduction, distinct from simply reusing the flat
port's equal-shard-size approach.
"""
import math

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import get_optimizer
from .boundary_optimizer import BoundaryOptimizer
from .losses import masked_sq_sum, reduced_masked_mse_loss, residual_targets


def is_main_process():
    return (not dist.is_initialized()) or dist.get_rank() == 0


class PI3NNConvTrainer:
    def __init__(
        self, configs, net_mean, net_up, net_down,
        train_dl, valid_dl, train_dl_full,
        train_sampler=None, device='cpu',
    ):
        """
        train_dl/valid_dl: used for the per-epoch SGD training/validation
        loop. May be built from a DistributedSampler (sharded per rank) --
        validation loss is computed correctly regardless, by all_reducing
        the per-rank sq_sum/count before dividing (see _run_valid_epoch),
        the same trick losses.reduced_masked_mse_loss uses for training.

        train_dl_full: a SEPARATE, non-distributed (no sampler) DataLoader
        over the complete, unsharded training set. boundary_optimization()
        needs actual per-pixel values for its bisection search, which
        isn't reducible to a sum/count the way MSE is -- so it does one
        rank-0-only full pass over this loader instead. For a
        single-process run (validate_synthetic.py), just pass the same
        DataLoader as train_dl here.
        """
        self.configs = configs
        self.device = device
        self.net_mean = net_mean.to(device)
        self.net_up = net_up.to(device)
        self.net_down = net_down.to(device)
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.train_dl_full = train_dl_full
        self.train_sampler = train_sampler
        self.c_up = None
        self.c_down = None

        # DDP only ever wraps the forward pass used for a phase's gradient
        # step (see _train_phase) -- boundary_optimization() and
        # evaluate() always call self.net_mean/up/down (the raw, unwrapped
        # modules) directly, since those are forward-only computations.
        self._ddp_mean = self._ddp_up = self._ddp_down = None
        if dist.is_initialized():
            device_ids = [torch.device(device).index] if torch.device(device).type == 'cuda' else None
            self._ddp_mean = DDP(self.net_mean, device_ids=device_ids)
            self._ddp_up = DDP(self.net_up, device_ids=device_ids)
            self._ddp_down = DDP(self.net_down, device_ids=device_ids)

    def _prepare_batch(self, batch):
        """Moves a batch to device and scales it in place, matching
        train.py's train_epoch convention exactly (raw_model.scale_pressure
        (state)/scale_evaptrans(evaptrans)/scale_statics(params)/
        scale_pressure(y) before any forward pass). net_mean/net_up/
        net_down all share the same scalers dict (same model_def passed to
        all three in build_networks), so scaling once via net_mean's
        methods is correct and equivalent to using net_up's or net_down's."""
        state, evaptrans, params, y = (t.to(self.device) for t in batch)
        self.net_mean.scale_pressure(state)
        self.net_mean.scale_evaptrans(evaptrans)
        self.net_mean.scale_statics(params)
        self.net_mean.scale_pressure(y)
        return state, evaptrans, params, y

    def _run_valid_epoch(self, model, predict_fn):
        model.eval()
        sq_sum = torch.zeros((), device=self.device)
        count = torch.zeros((), device=self.device)
        with torch.no_grad():
            for batch in self.valid_dl:
                state, evaptrans, params, y = self._prepare_batch(batch)
                pred, target, mask = predict_fn(model, state, evaptrans, params, y)
                s, c = masked_sq_sum(pred, target, mask)
                sq_sum += s
                count += c
        if dist.is_initialized():
            dist.all_reduce(sq_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
        return (sq_sum / count.clamp_min(1)).item()

    def _train_phase(self, model, ddp_model, predict_fn, label):
        cfg = self.configs
        optimizer = get_optimizer(cfg['optimizers'][label], model, cfg['lr'][label])
        forward_model = ddp_model if ddp_model is not None else model
        max_epochs = cfg['max_epochs'][label]
        wait_patience = cfg['wait_patience']
        early_stop_start_epoch = cfg.get('early_stop_start_epoch', 0)
        restore_best_weights = cfg.get('restore_best_weights', True)
        early_stop = cfg.get('early_stop', True)
        verbose = cfg.get('verbose', 0)

        best_loss = math.inf
        best_state = None
        wait = 0

        for epoch in range(max_epochs):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            model.train()
            for batch in self.train_dl:
                state, evaptrans, params, y = self._prepare_batch(batch)
                optimizer.zero_grad()
                pred, target, mask = predict_fn(forward_model, state, evaptrans, params, y)
                loss = reduced_masked_mse_loss(pred, target, mask)
                loss.backward()
                optimizer.step()

            valid_loss = self._run_valid_epoch(model, predict_fn)

            if verbose > 0 and is_main_process():
                print(f'[{label}] epoch {epoch}, valid_loss {valid_loss:.4e}')

            if early_stop and epoch >= early_stop_start_epoch:
                if dist.is_initialized():
                    # Broadcast rank 0's decision every epoch so no rank's
                    # best_loss/wait/stop bookkeeping can diverge from
                    # rank 0's, regardless of per-GPU floating-point
                    # non-determinism -- identical mechanism to the flat
                    # port's _train_one_network.
                    if dist.get_rank() == 0:
                        if valid_loss < best_loss:
                            new_best_loss, new_wait, improved = valid_loss, 0, 1.0
                        else:
                            new_best_loss, new_wait, improved = best_loss, wait + 1, 0.0
                        do_stop = 1.0 if new_wait >= wait_patience else 0.0
                    else:
                        new_best_loss, new_wait, improved, do_stop = 0.0, 0.0, 0.0, 0.0
                    state_t = torch.tensor(
                        [new_best_loss, float(new_wait), improved, do_stop],
                        dtype=torch.float64, device=self.device,
                    )
                    dist.broadcast(state_t, src=0)
                    best_loss, wait_f, improved, do_stop = state_t.tolist()
                    wait = int(wait_f)
                    if improved and restore_best_weights:
                        best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    if do_stop:
                        if restore_best_weights and best_state is not None:
                            model.load_state_dict(best_state)
                        if verbose > 0 and is_main_process():
                            print(f'[{label}] early stop at epoch {epoch}')
                        break
                else:
                    if valid_loss < best_loss:
                        best_loss = valid_loss
                        wait = 0
                        if restore_best_weights:
                            best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    else:
                        wait += 1
                        if wait >= wait_patience:
                            if restore_best_weights and best_state is not None:
                                model.load_state_dict(best_state)
                            if verbose > 0:
                                print(f'[{label}] early stop at epoch {epoch}')
                            break

    def train(self):
        def mean_predict(fm, state, evaptrans, params, y):
            return fm(state, evaptrans, params), y, torch.ones_like(y, dtype=torch.bool)

        self._train_phase(self.net_mean, self._ddp_mean, mean_predict, label='mean')

        # Only after net_mean is fully trained: residual_targets always
        # calls it under no_grad, so no explicit freezing/requires_grad_
        # is needed -- net_mean's parameters never receive gradients
        # during the up/down phases below regardless.
        self.net_mean.eval()

        def up_predict(fm, state, evaptrans, params, y):
            up_target, up_mask, _, _ = residual_targets(self.net_mean, state, evaptrans, params, y)
            return fm(state, evaptrans, params), up_target, up_mask

        self._train_phase(self.net_up, self._ddp_up, up_predict, label='up')

        def down_predict(fm, state, evaptrans, params, y):
            _, _, down_target, down_mask = residual_targets(self.net_mean, state, evaptrans, params, y)
            return fm(state, evaptrans, params), down_target, down_mask

        self._train_phase(self.net_down, self._ddp_down, down_predict, label='down')

    def boundary_optimization(self, verbose=0):
        self.net_mean.eval()
        self.net_up.eval()
        self.net_down.eval()
        out_channels = self.net_mean.output_channels

        c_up = torch.zeros(out_channels, dtype=torch.float64)
        c_down = torch.zeros(out_channels, dtype=torch.float64)

        if is_main_process():
            means, ups, downs, ys = [], [], [], []
            with torch.no_grad():
                for batch in self.train_dl_full:
                    state, evaptrans, params, y = self._prepare_batch(batch)
                    means.append(self.net_mean(state, evaptrans, params).cpu())
                    ups.append(self.net_up(state, evaptrans, params).cpu())
                    downs.append(self.net_down(state, evaptrans, params).cpu())
                    ys.append(y.cpu())
            mean_t = torch.cat(means)
            up_t = torch.cat(ups)
            down_t = torch.cat(downs)
            y_t = torch.cat(ys)

            quantile = self.configs['quantile']
            for c in range(out_channels):
                y_c = y_t[:, c].flatten().numpy()
                m_c = mean_t[:, c].flatten().numpy()
                u_c = up_t[:, c].flatten().numpy()
                d_c = down_t[:, c].flatten().numpy()
                num_outlier = int(y_c.shape[0] * (1 - quantile) / 2)
                opt = BoundaryOptimizer(
                    y_c, m_c, u_c, d_c, num_outlier=num_outlier,
                    c_up0_ini=0.0, c_up1_ini=100000.0,
                    c_down0_ini=0.0, c_down1_ini=100000.0, max_iter=1000,
                )
                c_up[c] = opt.optimize_up(verbose=verbose)
                c_down[c] = opt.optimize_down(verbose=verbose)
                if verbose > 0:
                    print(f'[channel {c}] c_up: {c_up[c].item():.4f}, c_down: {c_down[c].item():.4f}')

        if dist.is_initialized():
            c_up = c_up.to(self.device)
            c_down = c_down.to(self.device)
            dist.broadcast(c_up, src=0)
            dist.broadcast(c_down, src=0)
            c_up = c_up.cpu()
            c_down = c_down.cpu()

        self.c_up = c_up
        self.c_down = c_down

    def _caps_over_loader(self, loader, reduce_across_ranks):
        out_channels = self.net_mean.output_channels
        sq_err_sum = torch.zeros(out_channels, device=self.device)
        inside_sum = torch.zeros(out_channels, device=self.device)
        width_sum = torch.zeros(out_channels, device=self.device)
        y_sum = torch.zeros(out_channels, device=self.device)
        y_sq_sum = torch.zeros(out_channels, device=self.device)
        count = torch.zeros(out_channels, device=self.device)

        c_up = self.c_up.to(self.device).view(1, -1, 1, 1)
        c_down = self.c_down.to(self.device).view(1, -1, 1, 1)

        with torch.no_grad():
            for batch in loader:
                state, evaptrans, params, y = self._prepare_batch(batch)
                mean_pred = self.net_mean(state, evaptrans, params)
                up_pred = self.net_up(state, evaptrans, params)
                down_pred = self.net_down(state, evaptrans, params)
                upper = mean_pred + c_up * up_pred
                lower = mean_pred - c_down * down_pred
                inside = ((y <= upper) & (y >= lower)).to(y.dtype)
                width = upper - lower
                sq_err = (mean_pred - y) ** 2
                for c in range(out_channels):
                    sq_err_sum[c] += sq_err[:, c].sum()
                    inside_sum[c] += inside[:, c].sum()
                    width_sum[c] += width[:, c].sum()
                    y_sum[c] += y[:, c].sum()
                    y_sq_sum[c] += (y[:, c] ** 2).sum()
                    count[c] += inside[:, c].numel()

        if reduce_across_ranks and dist.is_initialized():
            for t in (sq_err_sum, inside_sum, width_sum, y_sum, y_sq_sum, count):
                dist.all_reduce(t, op=dist.ReduceOp.SUM)

        n = count.clamp_min(1)
        picp = (inside_sum / n).cpu()
        mpiw = (width_sum / n).cpu()
        rmse = (sq_err_sum / n).sqrt().cpu()
        ss_res = sq_err_sum
        ss_tot = y_sq_sum - (y_sum ** 2) / n
        r2 = (1 - ss_res / ss_tot.clamp_min(1e-12)).cpu()
        return {'picp': picp, 'mpiw': mpiw, 'rmse': rmse, 'r2': r2}

    def evaluate(self, verbose=0):
        """Returns per-channel (length out_channels) PICP/MPIW/RMSE/R2
        dicts for train (rank-0-only full pass, like
        boundary_optimization) and valid (every rank, exact via
        sum/count all_reduce over the possibly-sharded valid_dl)."""
        self.net_mean.eval()
        self.net_up.eval()
        self.net_down.eval()

        results = {}
        if is_main_process():
            results['train'] = self._caps_over_loader(self.train_dl_full, reduce_across_ranks=False)
        results['valid'] = self._caps_over_loader(self.valid_dl, reduce_across_ranks=True)

        if verbose > 0 and is_main_process():
            for split, metrics in results.items():
                for k, v in metrics.items():
                    print(f'{split}_{k}: {v.tolist()}')
        return results
