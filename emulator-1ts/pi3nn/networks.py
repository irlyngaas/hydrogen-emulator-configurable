"""
Conv2D PI3NN networks for emulator-1ts. Wraps model.py's existing
ResNet rather than reimplementing a convolutional architecture --
net_mean is just get_model('resnet', model_def) unmodified (true parity
with the regular single-model emulator); net_up/net_down wrap a second
and third ResNet instance with PI3NN's strictly-positive output
requirement.

v1 uses one shared, multi-output network per role: a single net_up and
a single net_down, each consuming the full input stack (same as
net_mean) and producing all out_channels in one forward pass. A
per-channel-network variant (one net_up_i/net_down_i pair per output
channel, same full input, output narrowed to 1 channel) was discussed
as a follow-up experiment -- trainer.py/losses.py/boundary_optimization
are written against the interface `forward(pressure, evaptrans, statics)
-> (N, C, H, W)` precisely so that swapping to the per-channel variant
later only means changing what build_networks() constructs here, not
touching those other files.
"""
import torch
import torch.nn as nn

from model import ResNet, get_model


class PositiveResNetWrapper(nn.Module):
    """
    Wraps a ResNet backbone with PI3NN's up/down-net requirements,
    mirroring UQNetStd.forward (UQnet/pi3nn_torch/networks.py) exactly,
    just swapping the flat Dense backbone for a Conv2D one:

    - A learnable bias, one scalar per output channel, initialized large
      (bias_init=3.0) -- this is the actual mechanism behind PI3NN's
      out-of-distribution awareness, not an incidental hyperparameter.
      Per-channel (not one global scalar) because different output
      channels (e.g. different pressure depth layers) can have very
      different residual scales, and boundary_optimization is already
      calibrated per-channel anyway -- see boundary_optimizer.py.
    - sqrt((x+bias)**2 + eps): guarantees a strictly positive output.
      BoundaryOptimizer's bisection search assumes mean +/- c*output
      moves monotonically as c increases, which requires this.

    eps=0.2 is kept matching the flat port's value, which was tuned for
    roughly-unit-variance standardized targets (same regime net_up/
    net_down operate in here, since ResNet's own scale_pressure
    standardizes the target). Worth a sanity check against real CONUS1
    residual magnitudes once real data is available.
    """

    def __init__(self, backbone: ResNet, bias_init=3.0, eps=0.2):
        super().__init__()
        self.backbone = backbone
        self.eps = eps
        self.custom_bias = nn.Parameter(
            torch.full((backbone.output_channels,), float(bias_init))
        )

    def forward(self, pressure, evaptrans, statics):
        x = self.backbone(pressure, evaptrans, statics)
        bias = self.custom_bias.view(1, -1, 1, 1)
        return torch.sqrt((x + bias) ** 2 + self.eps)

    # nn.Module.__getattr__ only forwards parameter/buffer/submodule
    # lookups, not arbitrary bound methods -- train_epoch's
    # raw_model.scale_pressure(...) (train.py:39-42) needs these to work
    # through the wrapper the same way they work on a bare ResNet.
    def scale_pressure(self, x):
        self.backbone.scale_pressure(x)

    def unscale_pressure(self, x):
        self.backbone.unscale_pressure(x)

    def scale_evaptrans(self, x):
        self.backbone.scale_evaptrans(x)

    def unscale_evaptrans(self, x):
        self.backbone.unscale_evaptrans(x)

    def scale_statics(self, x):
        self.backbone.scale_statics(x)

    def unscale_statics(self, x):
        self.backbone.unscale_statics(x)


def build_networks(model_def, bias_init=3.0, eps=0.2):
    """Returns (net_mean, net_up, net_down), each sharing the same
    model_def (in_channels/out_channels/hidden_dim/kernel_size/depth/
    scalers/pressure_names/evaptrans_names/param_names/n_evaptrans/
    parameter_list/param_nlayer), as independently-initialized ResNet
    instances -- net_mean unwrapped, net_up/net_down wrapped."""
    net_mean = get_model('resnet', model_def)
    net_up = PositiveResNetWrapper(get_model('resnet', model_def), bias_init, eps)
    net_down = PositiveResNetWrapper(get_model('resnet', model_def), bias_init, eps)
    return net_mean, net_up, net_down
