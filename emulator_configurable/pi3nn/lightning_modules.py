"""
The Lightning-facing side of PI3NN's up/down training. Registered via
the existing model_builder registry (@model_builder.register_emulator)
so it builds/trains/checkpoints through exactly the same path
ForcedSTRNN already does -- model_setup()'s build/device/precision/
learning_rate plumbing, train.py's pl.Trainer construction (SLURM/DDP
wiring, callbacks) -- all reused unchanged. See run_pi3nn_phase.py for
why this is still invoked once per phase (mean, then up, then down) via
separate train_model()/trainer.fit() calls, not via Lightning's
multi-optimizer support: PI3NN's phases are strictly sequential (train
mean fully before up/down even start), not alternating, which is what
Lightning's native multiple-optimizers mechanism is designed for
(GAN-style alternation) -- fighting that mismatch with
automatic_optimization=False would be more complex than just calling
fit() multiple times with an ordinary single-optimizer LightningModule
each time, which is what this class is.
"""
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from .. import model_builder
from .networks import build_updown_net, predict_updown
from .losses import residual_targets, reduced_masked_mse_loss


@model_builder.register_emulator('PI3NNUpDownModule')
class PI3NNUpDownModule(pl.LightningModule):
    def __init__(
        self, up_down_mode, role,
        mean_model_type, mean_model_config, mean_ckpt_path,
        updown_config,
    ):
        super().__init__()
        if role not in ('up', 'down'):
            raise ValueError(f"role must be 'up' or 'down', got {role!r}")
        self.up_down_mode = up_down_mode
        self.role = role

        self.net_mean = model_builder.ModelBuilder.build_emulator(mean_model_type, dict(mean_model_config))
        ckpt = torch.load(mean_ckpt_path, map_location='cpu')
        self.net_mean.load_state_dict(ckpt['state_dict'])
        self.net_mean.eval()
        for p in self.net_mean.parameters():
            p.requires_grad_(False)

        self.net = build_updown_net(up_down_mode, updown_config)

    def train(self, mode=True):
        # nn.Module.train() recursively sets ALL submodules to train
        # mode by default, which would silently undo net_mean.eval()
        # every time Lightning calls .train() on this whole module before
        # an epoch. For ForcedSTRNN specifically this happens to be
        # numerically harmless (Conv2d/LayerNorm2D are both mode-
        # insensitive -- no BatchNorm/Dropout anywhere), but pin it
        # explicitly rather than rely on that implicitly.
        super().train(mode)
        self.net_mean.eval()
        return self

    def forward(self, forcing, state, params):
        pred, _ = predict_updown(self.net_mean, self.net, self.up_down_mode, forcing, state, params)
        return pred

    def _step(self, batch):
        forcing, state, params, target = batch
        pred, mean_pred = predict_updown(self.net_mean, self.net, self.up_down_mode, forcing, state, params)
        up_target, up_mask, down_target, down_mask = residual_targets(target, mean_pred)
        tgt, mask = (up_target, up_mask) if self.role == 'up' else (down_target, down_mask)
        return reduced_masked_mse_loss(pred.squeeze(), tgt.squeeze(), mask.squeeze())

    def training_step(self, train_batch, train_batch_idx):
        loss = self._step(train_batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, val_batch_idx):
        loss = self._step(val_batch)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self, opt=torch.optim.AdamW):
        # self.net ONLY -- net_mean is frozen and excluded from the
        # optimizer entirely, on top of .eval()/requires_grad_(False)
        # above (belt-and-suspenders against it ever receiving a
        # gradient step).
        return opt(self.net.parameters(), lr=self.learning_rate, betas=[0.8, 0.95])

    def configure_loss(self, loss_fun=F.mse_loss):
        # model_setup() calls this unconditionally on whatever it builds
        # (see model_builder.py) -- accepted only for that API
        # compatibility. Not used: training_step/validation_step always
        # use reduced_masked_mse_loss, never a swappable loss_fun.
        pass
