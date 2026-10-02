"""
Runs the PyTorch PI3NN port on the boston-housing UCI dataset, using the
same hyperparameters and data split/scaling as UQnet's
`python main_PI3NN.py --data boston --mode manual --quantile 0.95`, so the
two can be compared directly.

Note on the scaling below -- replicated exactly from the TF original, not
"fixed": scalar_x.fit_transform() is called on xTrain, then called AGAIN on
xValid, which re-fits scalar_x's internal mean/std to xValid's statistics
and overwrites the fit from xTrain. xTest is then transformed using
whatever scalar_x's fit happens to be at that point (xValid's, due to the
overwrite) -- not xTrain's. Same for scalar_y. This is a real, odd quirk in
the original; reproducing it exactly is the point here, since this is a
parity check, not an improvement pass.

Exact numeric parity with the TF run isn't the bar -- TF and PyTorch use
different RNG streams and slightly different low-level Adam/initializer
implementations even with "the same" seed and hyperparameters, so some
run-to-run numeric difference is expected regardless of whether the port
is correct. What matters is whether PICP lands close to the target
quantile and RMSE/R2/MPIW are in the same ballpark as the TF reference run.
"""
import argparse

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .data import load_boston
from .networks import UQNetMean, UQNetStd
from .trainer import PI3NNTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quantile', type=float, default=0.95)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--verbose', type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    X, Y = load_boston()
    Y = Y.reshape(-1, 1)

    x_train_valid, x_test, y_train_valid, y_test = train_test_split(
        X, Y, test_size=0.1, random_state=1, shuffle=True
    )
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train_valid, y_train_valid, test_size=0.1, random_state=1, shuffle=True
    )

    # See module docstring -- this double fit_transform is a faithfully
    # reproduced quirk from the TF original, not a bug introduced here.
    scalar_x = StandardScaler()
    scalar_y = StandardScaler()
    x_train = scalar_x.fit_transform(x_train)
    x_valid = scalar_x.fit_transform(x_valid)
    x_test = scalar_x.transform(x_test)
    y_train = scalar_y.fit_transform(y_train)
    y_valid = scalar_y.fit_transform(y_valid)
    y_test = scalar_y.transform(y_test)

    num_inputs = x_train.shape[1]
    num_outputs = 1

    configs = {
        'quantile': args.quantile,
        'verbose': args.verbose,
        'Max_iter': 5000,
        'lr': [0.02, 0.02, 0.02],
        'optimizers': ['Adam', 'Adam', 'Adam'],
        'exponential_decay': True,
        'decay_steps': 3000,
        'decay_rate': 0.9,
        'early_stop': True,
        'early_stop_start_iter': 100,
        'wait_patience': 300,
        'restore_best_weights': True,
    }

    net_mean = UQNetMean(num_inputs, num_outputs, num_neurons=[50])
    net_up = UQNetStd(num_inputs, num_outputs, num_neurons=[50])
    net_down = UQNetStd(num_inputs, num_outputs, num_neurons=[50])

    trainer = PI3NNTrainer(
        configs, net_mean, net_up, net_down,
        x_train, y_train, x_valid, y_valid, x_test, y_test,
    )
    trainer.train()
    trainer.boundary_optimization(verbose=args.verbose)
    results = trainer.evaluate(final_evaluation=True, verbose=0)

    print('-' * 40)
    print(f'Quantile target: {args.quantile}')
    for k, v in results.items():
        print(f'{k}: {v:.4f}')
    print('-' * 40)
    print('Compare against: python main_PI3NN.py --data boston --mode manual --quantile', args.quantile)


if __name__ == '__main__':
    main()
