"""
Vendored verbatim from /home/i0l/UQnet/pi3nn_torch/boundary_optimizer.py
(the validated PyTorch PI3NN port -- numerically checked against the
original TF implementation on boston-housing, see that repo's
pi3nn_torch/validation/README.md). Copied rather than imported across
repos because a runtime dependency on UQnet/ being checked out at some
assumed path would be fragile on Frontier. If a bug is ever found here,
fix it in both places -- this is a point-in-time copy, not a live link.

Bisection search for scalar c_up/c_down such that exactly `num_outlier`
training points fall outside mean +/- c*output. Relies on output_up/
output_down being strictly positive (see networks.py's
PositiveResNetWrapper) for c*output to move the bound monotonically as c
increases -- without that, this search isn't well-posed.

Pure numpy, operates on flattened 1D arrays -- dimension-agnostic. For
the Conv2D/multi-channel case, call this once per output channel (see
trainer.py's boundary_optimization()), not once over every channel
pooled together: pooling would calibrate to the aggregate coverage
across all channels rather than each channel's own coverage, which
defeats the purpose of per-channel calibration.
"""
import numpy as np


def _to_numpy(x):
    if hasattr(x, 'detach'):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class BoundaryOptimizer:
    def __init__(self, y_train, output_mean, output_up, output_down,
                 num_outlier=None,
                 c_up0_ini=0.0, c_up1_ini=100000.0,
                 c_down0_ini=0.0, c_down1_ini=100000.0,
                 max_iter=1000):
        self.y_train = _to_numpy(y_train).flatten()
        self.output_mean = _to_numpy(output_mean).flatten()
        self.output_up = _to_numpy(output_up).flatten()
        self.output_down = _to_numpy(output_down).flatten()
        self.num_outlier = num_outlier
        self.c_up0_ini = c_up0_ini
        self.c_up1_ini = c_up1_ini
        self.c_down0_ini = c_down0_ini
        self.c_down1_ini = c_down1_ini
        self.max_iter = max_iter

    def optimize_up(self, outliers=None, verbose=0):
        if outliers is not None:
            self.num_outlier = outliers
        c_up0 = self.c_up0_ini
        c_up1 = self.c_up1_ini
        f0 = np.count_nonzero(self.y_train >= self.output_mean + c_up0 * self.output_up) - self.num_outlier
        f1 = np.count_nonzero(self.y_train >= self.output_mean + c_up1 * self.output_up) - self.num_outlier

        c_up2 = c_up1
        it = 0
        while it <= self.max_iter and f0 != 0 and f1 != 0:
            c_up2 = (c_up0 + c_up1) / 2.0
            f2 = np.count_nonzero(self.y_train >= self.output_mean + c_up2 * self.output_up) - self.num_outlier
            if f2 == 0:
                break
            elif f2 > 0:
                c_up0 = c_up2
                f0 = f2
            else:
                c_up1 = c_up2
                f1 = f2
            it += 1
            if verbose > 1:
                print(f'{it}, f0: {f0}, f1: {f1}, f2: {f2}')
        return c_up2

    def optimize_down(self, outliers=None, verbose=0):
        if outliers is not None:
            self.num_outlier = outliers
        c_down0 = self.c_down0_ini
        c_down1 = self.c_down1_ini
        f0 = np.count_nonzero(self.y_train <= self.output_mean - c_down0 * self.output_down) - self.num_outlier
        f1 = np.count_nonzero(self.y_train <= self.output_mean - c_down1 * self.output_down) - self.num_outlier

        c_down2 = c_down1
        it = 0
        while it <= self.max_iter and f0 != 0 and f1 != 0:
            c_down2 = (c_down0 + c_down1) / 2.0
            f2 = np.count_nonzero(self.y_train <= self.output_mean - c_down2 * self.output_down) - self.num_outlier
            if f2 == 0:
                break
            elif f2 > 0:
                c_down0 = c_down2
                f0 = f2
            else:
                c_down1 = c_down2
                f1 = f2
            it += 1
            if verbose > 1:
                print(f'{it}, f0: {f0}, f1: {f1}, f2: {f2}')
        return c_down2
