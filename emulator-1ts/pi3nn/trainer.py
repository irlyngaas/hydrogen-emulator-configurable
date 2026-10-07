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

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import get_optimizer
from .boundary_optimizer import BoundaryOptimizer
from .losses import masked_sq_sum, reduced_masked_mse_loss, residual_targets
from .spatial_calibration import fit_spatial_field


def is_main_process():
    return (not dist.is_initialized()) or dist.get_rank() == 0


class PI3NNConvTrainer:
    def __init__(
        self, configs, net_mean, net_up, net_down,
        train_dl, valid_dl, train_dl_full,
        train_sampler=None, device='cpu',
        train_dl_full_coords=None, valid_dl_coords=None,
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

        train_dl_full_coords/valid_dl_coords: optional, only needed for
        boundary_optimization_spatial_field()/evaluate_spatial_field()'s
        coords_mode='absolute' -- DataLoaders built from
        ParFlowDataset(..., return_coords=True) (split='train'/'valid'
        respectively), so each batch also carries this sample's absolute
        (y_min, x_min) position in the domain grid. Not needed for
        coords_mode='patch_relative' (the default), which reuses
        train_dl_full/valid_dl exactly like the scalar methods do, since
        position-within-patch needs no coordinate plumbing at all.
        """
        self.configs = configs
        self.device = device
        self.net_mean = net_mean.to(device)
        self.net_up = net_up.to(device)
        self.net_down = net_down.to(device)
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.train_dl_full = train_dl_full
        self.train_dl_full_coords = train_dl_full_coords
        self.valid_dl_coords = valid_dl_coords
        self.train_sampler = train_sampler
        self.c_up = None
        self.c_down = None
        self.c_up_field = None
        self.c_down_field = None
        self.spatial_field_coords_mode = None

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

    def boundary_optimization_spatial_field(self, coords_mode='patch_relative', rank=3, verbose=0):
        """Spatial-field analog of boundary_optimization(): c_up/c_down
        become a smoothed per-cell field instead of one scalar per
        channel. See pi3nn/spatial_calibration.py's module docstring for
        the three-stage algorithm (per-cell BoundaryOptimizer fit -> DCT
        low-rank smoothing -> global rescale).

        coords_mode: 'patch_relative' (default) -- field indexed by
        position within each (patch_size, patch_size) sample, reuses
        self.train_dl_full exactly like boundary_optimization() does, no
        dataset changes needed. 'absolute' -- field indexed by this
        sample's real position in the fixed CONUS1 domain grid (same
        grid/size every call, since ParFlowDataset's xbatcher
        BatchGenerator tiles the same fixed extent every time) -- needs
        self.train_dl_full_coords (see __init__'s docstring)."""
        self.net_mean.eval()
        self.net_up.eval()
        self.net_down.eval()
        out_channels = self.net_mean.output_channels

        if coords_mode == 'absolute' and self.train_dl_full_coords is None:
            raise ValueError(
                "coords_mode='absolute' needs train_dl_full_coords (a DataLoader built from "
                "ParFlowDataset(..., return_coords=True)) passed to PI3NNConvTrainer's constructor."
            )
        loader = self.train_dl_full_coords if coords_mode == 'absolute' else self.train_dl_full
        ds = loader.dataset
        field_h, field_w = (ds.Y_EXTENT, ds.X_EXTENT) if coords_mode == 'absolute' else (ds.patch_size, ds.patch_size)
        n_cells = field_h * field_w

        c_up_field = torch.zeros(out_channels, field_h, field_w, dtype=torch.float64)
        c_down_field = torch.zeros(out_channels, field_h, field_w, dtype=torch.float64)

        if is_main_process():
            means, ups, downs, ys, coords = [], [], [], [], []
            with torch.no_grad():
                for batch in loader:
                    if coords_mode == 'absolute':
                        state, evaptrans, params, y, coord = batch
                        coords.append(coord)
                    else:
                        state, evaptrans, params, y = batch
                    state, evaptrans, params, y = self._prepare_batch((state, evaptrans, params, y))
                    means.append(self.net_mean(state, evaptrans, params).cpu())
                    ups.append(self.net_up(state, evaptrans, params).cpu())
                    downs.append(self.net_down(state, evaptrans, params).cpu())
                    ys.append(y.cpu())
            mean_t = torch.cat(means)
            up_t = torch.cat(ups)
            down_t = torch.cat(downs)
            y_t = torch.cat(ys)
            n_sample, _, patch_h, patch_w = mean_t.shape

            local_h = np.arange(patch_h)[:, None]
            local_w = np.arange(patch_w)[None, :]
            if coords_mode == 'absolute':
                coord_t = torch.cat(coords).numpy()
                global_h = coord_t[:, 0][:, None, None] + local_h[None, :, :]
                global_w = coord_t[:, 1][:, None, None] + local_w[None, :, :]
                global_h = np.broadcast_to(global_h, (n_sample, patch_h, patch_w))
                global_w = np.broadcast_to(global_w, (n_sample, patch_h, patch_w))
                cell_hw = (global_h * field_w + global_w).astype(np.int64)
            else:
                cell_hw = np.broadcast_to((local_h * patch_w + local_w)[None, :, :], (n_sample, patch_h, patch_w)).astype(np.int64)
            cell_index = cell_hw.reshape(-1)

            quantile = self.configs['quantile']
            for c in range(out_channels):
                y_c = y_t[:, c].numpy().reshape(-1)
                m_c = mean_t[:, c].numpy().reshape(-1)
                u_c = up_t[:, c].numpy().reshape(-1)
                d_c = down_t[:, c].numpy().reshape(-1)
                c_up_f, c_down_f, alpha_up, alpha_down = fit_spatial_field(
                    y_c, m_c, u_c, d_c, cell_index, n_cells, (field_h, field_w), quantile, rank,
                )
                c_up_field[c] = torch.from_numpy(c_up_f)
                c_down_field[c] = torch.from_numpy(c_down_f)
                if verbose > 0:
                    print(f'[channel {c}] alpha_up: {alpha_up:.4f}, alpha_down: {alpha_down:.4f}')

        if dist.is_initialized():
            c_up_field = c_up_field.to(self.device)
            c_down_field = c_down_field.to(self.device)
            dist.broadcast(c_up_field, src=0)
            dist.broadcast(c_down_field, src=0)
            c_up_field = c_up_field.cpu()
            c_down_field = c_down_field.cpu()

        self.c_up_field = c_up_field
        self.c_down_field = c_down_field
        self.spatial_field_coords_mode = coords_mode

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

    def _caps_over_loader_spatial_field(self, loader, coords_mode, reduce_across_ranks):
        out_channels = self.net_mean.output_channels
        sq_err_sum = torch.zeros(out_channels, device=self.device)
        inside_sum = torch.zeros(out_channels, device=self.device)
        width_sum = torch.zeros(out_channels, device=self.device)
        y_sum = torch.zeros(out_channels, device=self.device)
        y_sq_sum = torch.zeros(out_channels, device=self.device)
        count = torch.zeros(out_channels, device=self.device)

        c_up_field = self.c_up_field.numpy()    # (out_channels, field_h, field_w)
        c_down_field = self.c_down_field.numpy()
        field_h, field_w = self.c_up_field.shape[1], self.c_up_field.shape[2]
        n_cells = field_h * field_w
        # Per-cell accumulation (same spirit as _per_cell_picp_stats in
        # spatial_calibration.py, but streamed across this loader's batches
        # via scatter_add_ instead of needing every point materialized in
        # memory at once) -- the pooled picp/mpiw/rmse/r2 below hide
        # whether the field is actually locally well-calibrated everywhere
        # or just correct on (pooled) average; this is the diagnostic that
        # tells the difference, same one calibrate_spatial_field() already
        # reports in emulator_configurable.
        per_cell_inside_sum = torch.zeros(out_channels, n_cells, device=self.device)
        per_cell_count = torch.zeros(out_channels, n_cells, device=self.device)

        with torch.no_grad():
            for batch in loader:
                if coords_mode == 'absolute':
                    state, evaptrans, params, y, coord = batch
                    coord = coord.numpy()
                else:
                    state, evaptrans, params, y = batch
                    coord = None
                state, evaptrans, params, y = self._prepare_batch((state, evaptrans, params, y))
                mean_pred = self.net_mean(state, evaptrans, params)
                up_pred = self.net_up(state, evaptrans, params)
                down_pred = self.net_down(state, evaptrans, params)

                batch_n, _, patch_h, patch_w = mean_pred.shape
                local_h = np.arange(patch_h)[:, None]
                local_w = np.arange(patch_w)[None, :]
                if coords_mode == 'absolute':
                    global_h = coord[:, 0][:, None, None] + local_h[None, :, :]
                    global_w = coord[:, 1][:, None, None] + local_w[None, :, :]
                    global_h = np.broadcast_to(global_h, (batch_n, patch_h, patch_w))
                    global_w = np.broadcast_to(global_w, (batch_n, patch_h, patch_w))
                else:
                    global_h = np.broadcast_to(local_h[None, :, :], (batch_n, patch_h, patch_w))
                    global_w = np.broadcast_to(local_w[None, :, :], (batch_n, patch_h, patch_w))

                # c_up_pp[c, n, h, w] = c_up_field[c, global_h[n,h,w], global_w[n,h,w]]
                # -- fancy-indexed in numpy (simpler to get right than torch's
                # advanced-indexing broadcast rules for this exact gather),
                # then moved to this batch's device/dtype for the arithmetic.
                c_up_pp = c_up_field[:, global_h, global_w]
                c_down_pp = c_down_field[:, global_h, global_w]
                c_up_pp = torch.from_numpy(c_up_pp).to(mean_pred.dtype).to(self.device).permute(1, 0, 2, 3)
                c_down_pp = torch.from_numpy(c_down_pp).to(mean_pred.dtype).to(self.device).permute(1, 0, 2, 3)

                upper = mean_pred + c_up_pp * up_pred
                lower = mean_pred - c_down_pp * down_pred
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

                cell_idx_batch = torch.from_numpy((global_h * field_w + global_w).astype(np.int64)).to(self.device)
                cell_idx_exp = cell_idx_batch.unsqueeze(1).expand(-1, out_channels, -1, -1)
                chan_idx_exp = torch.arange(out_channels, device=self.device).view(1, -1, 1, 1).expand(batch_n, -1, patch_h, patch_w)
                combined_idx = (chan_idx_exp * n_cells + cell_idx_exp).reshape(-1)
                per_cell_inside_sum.view(-1).scatter_add_(0, combined_idx, inside.reshape(-1).to(per_cell_inside_sum.dtype))
                per_cell_count.view(-1).scatter_add_(0, combined_idx, torch.ones_like(inside.reshape(-1)).to(per_cell_count.dtype))

        if reduce_across_ranks and dist.is_initialized():
            for t in (sq_err_sum, inside_sum, width_sum, y_sum, y_sq_sum, count, per_cell_inside_sum, per_cell_count):
                dist.all_reduce(t, op=dist.ReduceOp.SUM)

        n = count.clamp_min(1)
        picp = (inside_sum / n).cpu()
        mpiw = (width_sum / n).cpu()
        rmse = (sq_err_sum / n).sqrt().cpu()
        ss_res = sq_err_sum
        ss_tot = y_sq_sum - (y_sum ** 2) / n
        r2 = (1 - ss_res / ss_tot.clamp_min(1e-12)).cpu()

        per_cell_picp = (per_cell_inside_sum / per_cell_count.clamp_min(1)).cpu().numpy()
        valid_cell = (per_cell_count > 0).cpu().numpy()
        picp_spatial_mean = torch.full((out_channels,), float('nan'))
        picp_spatial_std = torch.full((out_channels,), float('nan'))
        picp_spatial_min = torch.full((out_channels,), float('nan'))
        picp_spatial_max = torch.full((out_channels,), float('nan'))
        for c in range(out_channels):
            vals = per_cell_picp[c][valid_cell[c]]
            if vals.size:
                picp_spatial_mean[c] = float(vals.mean())
                picp_spatial_std[c] = float(vals.std())
                picp_spatial_min[c] = float(vals.min())
                picp_spatial_max[c] = float(vals.max())

        return {
            'picp': picp, 'mpiw': mpiw, 'rmse': rmse, 'r2': r2,
            'picp_spatial_mean': picp_spatial_mean, 'picp_spatial_std': picp_spatial_std,
            'picp_spatial_min': picp_spatial_min, 'picp_spatial_max': picp_spatial_max,
        }

    def evaluate_spatial_field(self, verbose=0):
        """Spatial-field analog of evaluate(): per-channel PICP/MPIW/
        RMSE/R2 for train (rank-0-only full pass) and valid (every rank,
        reduced), using self.c_up_field/self.c_down_field instead of
        scalar c_up/c_down. Must be called after
        boundary_optimization_spatial_field() (reads
        self.spatial_field_coords_mode to know which loaders to use)."""
        if self.c_up_field is None:
            raise RuntimeError('evaluate_spatial_field() called before boundary_optimization_spatial_field()')
        coords_mode = self.spatial_field_coords_mode
        if coords_mode == 'absolute' and self.valid_dl_coords is None:
            raise ValueError(
                "coords_mode='absolute' needs valid_dl_coords (a DataLoader built from "
                "ParFlowDataset(..., return_coords=True, split='valid')) passed to "
                "PI3NNConvTrainer's constructor to evaluate the valid split."
            )
        self.net_mean.eval()
        self.net_up.eval()
        self.net_down.eval()

        results = {}
        if is_main_process():
            train_loader = self.train_dl_full_coords if coords_mode == 'absolute' else self.train_dl_full
            results['train'] = self._caps_over_loader_spatial_field(train_loader, coords_mode, reduce_across_ranks=False)
        valid_loader = self.valid_dl_coords if coords_mode == 'absolute' else self.valid_dl
        results['valid'] = self._caps_over_loader_spatial_field(valid_loader, coords_mode, reduce_across_ranks=True)

        if verbose > 0 and is_main_process():
            for split, metrics in results.items():
                for k, v in metrics.items():
                    print(f'{split}_{k}: {v.tolist()}')
        return results
