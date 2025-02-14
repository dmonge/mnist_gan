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

from generator import Generator
from discriminator import Discriminator
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
plt.show()

# models
generator = Generator(latent_size, n_classes, hidden_size, img_size)
discriminator = Discriminator(img_size, hidden_size)
print(generator)
print(discriminator)

# train
criterion = BCELoss()
g_optimizer = Adam(generator.parameters(), lr=lr)
d_optimizer = Adam(discriminator.parameters(), lr=lr)
losses = []
for epoch in range(n_epochs):

    epoch_loss = 0
    n_steps = 1
    for step, (x_train, label) in enumerate(train_dataloader):

        # generation
        labels = torch.randint(0, n_classes, (batch_size, ))
        latent_vector = torch.rand((batch_size, latent_size))
        latent_vector = torch.cat([latent_vector, F.one_hot(labels, num_classes=n_classes)], dim=1)
        x_generated = generator(latent_vector)

        # build batch: concat and shuffle
        x_train = torch.concat([x_train, x_generated])
        y_train = torch.concat([torch.full((batch_size, 1), REAL_IMAGE),
                                torch.full((batch_size, 1), FAKE_IMAGE)])
        _i = torch.randperm(batch_size * 2)

        # discrimination
        y_pred = discriminator(x_train)

        # loss
        g_loss = criterion(y_pred, torch.full((batch_size * 2, 1), REAL_IMAGE))
        d_loss = criterion(y_pred, y_train)
        loss = g_loss + d_loss
        epoch_loss += loss.detach().numpy()
        n_steps += 1

        losses.append((epoch, step, g_loss.detach().numpy(), d_loss.detach().numpy(), loss.detach().numpy()))
        if step % 100 == 0:
            print(f'({epoch} / {step}) {g_loss:0.3f} {d_loss:0.3f} {loss:0.3f}')

        # backward
        g_optimizer.zero_grad()
        d_optimizer.zero_grad()
        loss.backward()
        g_optimizer.step()
        g_optimizer.step()

    if epoch % 10 == 0:
        print(f'>> {epoch=} {epoch_loss / n_steps}')


# learning curves
_, _, gl, dl, l = zip(*losses)
plt.plot(l, label='loss')
plt.plot(gl, label='generator loss')
plt.plot(dl, label='discriminator loss')
plt.legend()
plt.show()


# generation
with torch.no_grad():
    n_images = 100
    labels = torch.randint(0, n_classes, (n_images, ))
    latent_vector = torch.rand((n_images, latent_size))
    latent_vector = torch.cat([latent_vector, F.one_hot(labels, num_classes=n_classes)], dim=1)
    x_generated = generator(latent_vector)

    grid = make_grid([x_generated[i] for i in range(n_images)], nrow=10)
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()
