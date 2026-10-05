"""
Three selectable architectures for PI3NN's net_up/net_down, all wrapping
or cloning emulator_configurable.models.ForcedSTRNN (net_mean's
architecture), plus the common call interface (predict_updown) the
training/calibration code uses so it doesn't need to know which variant
is active:

- stateless_output: a plain (non-recurrent) conv head per timestep,
  seeing [forcings_t, static, net_mean's frozen output_t].
- stateless_hidden: same, plus net_mean's own frozen per-timestep hidden
  state h_t (itself a spatial feature map, same H,W as everything else)
  as extra input channels -- captures whatever temporal signal
  net_mean's recurrence encoded, without net_up/net_down needing their
  own recurrent cell.
- recurrent_clone: a full independent ForcedSTRNN-shaped recurrent
  stack, mirroring net_mean's own architecture exactly (own hidden
  state, own autoregressive feedback).
"""
import torch
import torch.nn as nn

from ..models import ActionSTLSTMCell, BasicResNet


class PositiveBias(nn.Module):
    """PI3NN's per-channel large-init-bias + sqrt((x+bias)**2+eps)
    strictly-positive activation (see emulator-1ts/pi3nn/networks.py's
    PositiveResNetWrapper for the full rationale: bias_init=3.0 is the
    actual OOD-awareness mechanism, not an incidental hyperparameter;
    the sqrt form guarantees strict positivity, required for
    BoundaryOptimizer's monotonicity assumption).

    Factored into its own module, rather than inlined once per wrapper
    the way emulator-1ts's PositiveResNetWrapper does it, because
    RecurrentUpDownSTRNN needs to apply this INSIDE a per-timestep loop
    (on the value about to be fed forward as the next timestep's input),
    not as a post-hoc wrapper around a whole network's final output."""

    def __init__(self, channels, bias_init=3.0, eps=0.2):
        super().__init__()
        self.eps = eps
        self.custom_bias = nn.Parameter(torch.full((channels,), float(bias_init)))

    def forward(self, x):
        bias = self.custom_bias.view(1, -1, 1, 1)
        return torch.sqrt((x + bias) ** 2 + self.eps)


class StatelessUpDownNet(nn.Module):
    """Variants (a) stateless_output and (b) stateless_hidden. No
    recurrence of its own: every timestep is an independent prediction
    from [forcings_t, static, net_mean's frozen output_t] (+ net_mean's
    frozen hidden state h_t, if include_hidden). Since there's no
    cross-timestep dependency, this is vectorized over batch*timesteps
    in one shot rather than a Python per-timestep loop."""

    def __init__(
        self, act_channel, static_channel, out_channel,
        include_hidden=False, hidden_state_channel=0,
        hidden_dim=64, kernel_size=5, depth=1,
        bias_init=3.0, eps=0.2,
    ):
        super().__init__()
        self.include_hidden = include_hidden
        in_channels = act_channel + static_channel + out_channel
        if include_hidden:
            if hidden_state_channel <= 0:
                raise ValueError('include_hidden=True requires hidden_state_channel > 0')
            in_channels += hidden_state_channel
        self.backbone = BasicResNet(in_channels, out_channel, hidden_dim, kernel_size, depth)
        self.positive = PositiveBias(out_channel, bias_init, eps)

    def forward(self, forcings, static_inputs, mean_out, mean_hidden=None):
        batch, T, _, H, W = forcings.shape
        static = static_inputs[:, 0:1].expand(-1, T, -1, -1, -1)
        parts = [forcings, static, mean_out]
        if self.include_hidden:
            if mean_hidden is None:
                raise ValueError('include_hidden=True but mean_hidden was not provided')
            parts.append(mean_hidden)
        x = torch.cat(parts, dim=2).reshape(batch * T, -1, H, W)
        out = self.positive(self.backbone(x))
        return out.reshape(batch, T, -1, H, W)


class RecurrentUpDownSTRNN(nn.Module):
    """Variant (c): an independent ForcedSTRNN-shaped recurrent stack
    (own ActionSTLSTMCell layers, own hidden state) -- structurally
    identical to ForcedSTRNN.__init__/.forward, except PositiveBias is
    applied to the fed-forward value each timestep:
    `x = Positive(conv_last(h_t[-1]) + x)`, wrapping the WHOLE residual
    expression (confirmed design choice: matches net_mean's own update
    rule exactly, rather than replacing the residual entirely).

    Deliberately does not compute ForcedSTRNN's decouple_loss: that
    bookkeeping only ever gets consumed by configure_loss's loss_fun
    wrapper, and this class is a plain nn.Module (not a LightningModule)
    wrapped by lightning_modules.PI3NNUpDownModule, whose training step
    always uses reduced_masked_mse_loss directly -- there is no loss_fun
    here for decouple_loss to feed into."""

    def __init__(
        self, num_layers, num_hidden, act_channel, init_cond_channel,
        static_channel, out_channel, filter_size=5, stride=1,
        bias_init=3.0, eps=0.2,
    ):
        super().__init__()
        self.frame_channel = init_cond_channel
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.out_channel = out_channel

        cell_list = []
        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(ActionSTLSTMCell(in_channel, act_channel, num_hidden[i], filter_size, stride))
        self.cell_list = nn.ModuleList(cell_list)

        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], out_channel, kernel_size=1, bias=False)
        adapter_num_hidden = num_hidden[0]
        self.adapter = nn.Conv2d(adapter_num_hidden, adapter_num_hidden, kernel_size=1, bias=False)
        self.memory_encoder = nn.Conv2d(init_cond_channel, num_hidden[0], kernel_size=1, bias=True)
        self.cell_encoder = nn.Conv2d(static_channel, sum(num_hidden), kernel_size=1, bias=True)

        self.positive = PositiveBias(out_channel, bias_init, eps)

    def forward(self, forcings, init_cond, static_inputs):
        batch, timesteps, channels, height, width = forcings.shape

        next_frames = []
        h_t, c_t = [], []
        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width], device=forcings.device)
            h_t.append(zeros)
            c_t.append(zeros)

        memory = self.memory_encoder(init_cond[:, 0])
        c_t = list(torch.split(self.cell_encoder(static_inputs[:, 0]), self.num_hidden, dim=1))

        x = init_cond[:, 0]
        for t in range(timesteps):
            a = forcings[:, t]
            h_t[0], c_t[0], memory, _, _ = self.cell_list[0](x, a, h_t[0], c_t[0], memory)
            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory, _, _ = self.cell_list[i](h_t[i - 1], a, h_t[i], c_t[i], memory)

            x = self.positive(self.conv_last(h_t[-1]) + x)
            next_frames.append(x)

        return torch.stack(next_frames, dim=1)


def predict_updown(net_mean, net, up_down_mode, forcing, state, params):
    """Common call interface across all three variants. net_mean must
    already be frozen (.eval() + requires_grad_(False)); always called
    under no_grad here regardless. Returns (pred, mean_pred) --
    mean_pred is always computed (every variant's residual target/mask
    is defined against it, even recurrent_clone, which doesn't consume
    it as a forward input)."""
    if up_down_mode == 'stateless_hidden':
        with torch.no_grad():
            mean_pred, mean_hidden = net_mean(forcing, state, params, return_hidden=True)
        pred = net(forcing, params, mean_pred, mean_hidden)
    elif up_down_mode == 'stateless_output':
        with torch.no_grad():
            mean_pred = net_mean(forcing, state, params)
        pred = net(forcing, params, mean_pred)
    elif up_down_mode == 'recurrent_clone':
        with torch.no_grad():
            mean_pred = net_mean(forcing, state, params)
        pred = net(forcing, state, params)
    else:
        raise ValueError(f'unknown up_down_mode {up_down_mode!r}')
    return pred, mean_pred


def build_updown_net(up_down_mode, updown_config):
    if up_down_mode == 'stateless_output':
        return StatelessUpDownNet(include_hidden=False, **updown_config)
    elif up_down_mode == 'stateless_hidden':
        return StatelessUpDownNet(include_hidden=True, **updown_config)
    elif up_down_mode == 'recurrent_clone':
        return RecurrentUpDownSTRNN(**updown_config)
    raise ValueError(f'unknown up_down_mode {up_down_mode!r}')
