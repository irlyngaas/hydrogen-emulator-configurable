"""
PyTorch port of UQnet's pi3nn/Networks/networks.py (UQ_Net_mean_TF2,
UQ_Net_std_TF2). Ported for numeric/structural parity with the TF
original, not just "a 3-network UQ method" -- the specific details below
are methodologically load-bearing, not implementation incidentals (see the
conversation this was scoped from for why each one matters):

- UQNetStd's sqrt(x**2 + 0.2) output activation: guarantees a strictly
  positive output. The boundary optimizer's bisection search assumes
  mean +/- c*output moves monotonically as c increases, which requires
  this; without it the root-finding is invalid.
- custom_bias initialized to 3.0 (not 0): this is the actual mechanism
  behind PI3NN's out-of-distribution awareness. Intervals start wide
  everywhere and only shrink where training data justifies it -- drop
  this and you still have a PI3NN-shaped pipeline, but without its
  flagship feature.
- RandomNormal(mean=0.1, std=0.1) kernel init and L1+L2 regularization
  (applied only to the hidden fcs layers, not the input/output layers --
  matches the TF original's add_model_regularizer_loss, which only ever
  sees a kernel_regularizer on the fcs Dense layers) are incidental
  hyperparameter choices, not structurally required, but kept for
  numeric parity with the TF reference run.
"""
import torch
import torch.nn as nn


class UQNetMean(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_neurons):
        super().__init__()
        # Matches Dense(num_inputs, activation='linear') in the TF original --
        # a square linear map preserving dimensionality, not a typical
        # "input projection" layer. Kept as-is for parity, not because this
        # shape makes obvious sense on its own.
        self.input_layer = nn.Linear(num_inputs, num_inputs)

        layers = []
        in_dim = num_inputs
        for n in num_neurons:
            fc = nn.Linear(in_dim, n)
            nn.init.normal_(fc.weight, mean=0.1, std=0.1)
            layers.append(fc)
            in_dim = n
        self.fcs = nn.ModuleList(layers)
        self.output_layer = nn.Linear(in_dim, num_outputs)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.input_layer(x)
        for fc in self.fcs:
            x = self.activation(fc(x))
        x = self.output_layer(x)
        return x

    def regularizer_loss(self, l1=0.02, l2=0.02):
        """L1+L2 penalty on the hidden fcs layers' weights only -- matches
        add_model_regularizer_loss, which only ever sees a kernel_regularizer
        on those layers in the TF original (input_layer/output_layer were
        constructed with no regularizer kwarg)."""
        penalty = torch.zeros((), device=self.output_layer.weight.device)
        for fc in self.fcs:
            penalty = penalty + l1 * fc.weight.abs().sum() + l2 * fc.weight.pow(2).sum()
        return penalty


class UQNetStd(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_neurons, bias_init=3.0):
        super().__init__()
        self.input_layer = nn.Linear(num_inputs, num_inputs)

        layers = []
        in_dim = num_inputs
        for n in num_neurons:
            fc = nn.Linear(in_dim, n)
            nn.init.normal_(fc.weight, mean=0.1, std=0.1)
            layers.append(fc)
            in_dim = n
        self.fcs = nn.ModuleList(layers)
        self.output_layer = nn.Linear(in_dim, num_outputs)
        self.activation = nn.ReLU()

        # Matches tf.Variable([3.0]) -- a single scalar added to every output
        # unit, not one bias per output dimension. This large initial value,
        # not the fact that it's a bias at all, is what gives PI3NN its
        # OOD-awareness: intervals start wide everywhere and only shrink
        # where training data pulls this down during optimization.
        self.custom_bias = nn.Parameter(torch.tensor(float(bias_init)))

    def forward(self, x):
        x = self.input_layer(x)
        for fc in self.fcs:
            x = self.activation(fc(x))
        x = self.output_layer(x)
        x = x + self.custom_bias
        # Strictly positive by construction (sqrt(0.2) is the floor) --
        # required for the boundary optimizer's monotonicity assumption,
        # not just a numerically-convenient activation choice.
        x = torch.sqrt(x ** 2 + 0.2)
        return x

    def regularizer_loss(self, l1=0.02, l2=0.02):
        penalty = torch.zeros((), device=self.output_layer.weight.device)
        for fc in self.fcs:
            penalty = penalty + l1 * fc.weight.abs().sum() + l2 * fc.weight.pow(2).sum()
        return penalty
