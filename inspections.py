"""
Module with functions for training inspections.
"""
import torch

from matplotlib import pyplot as plt
from torchvision.utils import make_grid


def inspect_progress(epoch, x_train, discriminator):
    """Save inspections."""
    with torch.no_grad():
        n_batches, _, _, _ = x_train.shape

        # train batch
        grid = make_grid([normalize(x_train[i]) for i in range(x_train.size(0))], nrow=16)
        plt.imsave(f'inspections/batch_{epoch:06}.png', grid.permute(1, 2, 0).numpy())

        # predictions
        discriminator.eval()
        y_pred = discriminator(x_train)
        discriminator.train()
        grid = make_grid([tag_class(normalize(x_train[i]), y_pred[i] > 0) for i in range(x_train.size(0))], nrow=16)
        plt.imsave(f'inspections/predictions_{epoch:06}.png', grid.permute(1, 2, 0).numpy())


def tag_class(x, pred):
    """Tag real and fake images."""
    x = x.repeat((3, 1, 1))
    if pred:  # real: green
        x[1, -2:, -2:] = 1
    else:  # fake: red
        x[0, -2:, -2:] = 1
    return x


def normalize(x):
    """Go from [-1, 1] to [0, 1]."""
    return (x + 1) / 2
