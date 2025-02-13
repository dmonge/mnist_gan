"""
Train GAN.
"""
from torch.nn import BCELoss
from torch.optim import Adam
from torchvision.datasets import MNIST
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from matplotlib import pyplot as plt
from torchvision.transforms.v2 import ToTensor
from torchvision.utils import make_grid
from icecream import ic

from generator import Generator
from discriminator import Discriminator, REAL_IMAGE
from discriminator import REAL_IMAGE
from discriminator import FAKE_IMAGE


# parameters
n_epochs = 10
latent_size = 64
batch_size = 8
hidden_size = 16
lr = 1e-3
n_classes = 10

# data
train_data = MNIST('data', train=True, download=True, transform=ToTensor())
n_channels, img_size, _ = train_data[0][0].shape
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

grid = make_grid([train_data[i][0] for i in range(100)], nrow=10)
plt.imshow(grid.permute(1, 2, 0))
# plt.show()

# models
generator = Generator(latent_size, n_classes, hidden_size, img_size)
discriminator = Discriminator()
print(generator)
# print(discriminator)

# train
criterion = BCELoss()
g_optimizer = Adam(generator.parameters(), lr=lr)
d_optimizer = Adam(discriminator.parameters(), lr=lr)
for epoch in range(n_epochs):
    for x_train, label in train_dataloader:

        # generation
        labels = torch.randint(0, n_classes, (batch_size, ))
        latent_vector = torch.rand((batch_size, latent_size))
        latent_vector = torch.cat([latent_vector, F.one_hot(labels, num_classes=n_classes)], dim=1)
        x_generated = generator(latent_vector)

        # build batch: concat and shuffle
        x_train = torch.concat([x_train, x_generated])
        y_train = torch.concat([torch.full((batch_size, ), REAL_IMAGE),
                                torch.full((batch_size, ), FAKE_IMAGE)])
        _i = torch.randperm(batch_size * 2)
        ic(x_train.shape)
        ic(y_train.shape)

        # discrimination
        y_pred = discriminator(x_train, label)
        y_pred = (y_pred > 0.5).astype(int)

        # loss
        g_loss = criterion()
        d_loss = criterion()
        loss = g_loss + d_loss

        # backward
        g_optimizer.zero_grad(1 - y_pred, y_train)
        d_optimizer.zero_grad(y_pred, y_train)
        loss.backward()
        g_optimizer.step()
        g_optimizer.step()
