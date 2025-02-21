"""
Train GAN.
"""
import glob
import os

import numpy as np
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torchvision.datasets import MNIST
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from matplotlib import pyplot as plt
from torchvision.transforms.v2 import Compose
from torchvision.transforms.v2 import ToTensor
from torchvision.transforms.v2 import Normalize
from torchvision.utils import make_grid
from torchmetrics import Accuracy

from generator import Generator
from discriminator import Discriminator
from discriminator import REAL_IMAGE
from discriminator import FAKE_IMAGE
from inspections import inspect_progress


torch.manual_seed(7)

# parameters
n_epochs = 100
latent_size = 64
batch_size = 128
hidden_size = 64
lr_generator = 2e-3
lr_discriminator = 1e-3
n_classes = 10

# progress
progress_every = 100  # print progress every n steps
do_inspections = True  # save inspections?
inspections_every = 1  # save inspections every n epochs

# data
transform = Compose([
    ToTensor(),
    Normalize(mean=[0.5], std=[0.5])
])
train_data = MNIST('data', train=True, download=True, transform=transform)
n_channels, img_size, _ = train_data[0][0].shape
train_dataloader = DataLoader(train_data, batch_size=batch_size // 2, shuffle=True, drop_last=True,
                              num_workers=16, persistent_workers=True)
print('dataset size:', len(train_data))

# models
generator = Generator(latent_size, n_classes, hidden_size, img_size)
discriminator = Discriminator(img_size, hidden_size)

# inspections
os.makedirs('inspections', exist_ok=True)
for f in glob.glob('inspections/*'):
    os.remove(f)

# train
criterion = BCEWithLogitsLoss()
g_optimizer = Adam(generator.parameters(), lr=lr_generator)
d_optimizer = Adam(discriminator.parameters(), lr=lr_discriminator)
g_accuracy = Accuracy(task='binary')
d_accuracy = Accuracy(task='binary')
losses = []
accuracies = []

for epoch in range(n_epochs):
    epoch_loss = 0
    batch_losses = []
    batch_accuracies = []
    for step, (x_real, label) in enumerate(train_dataloader):
        # 0. build batch
        labels = torch.randint(0, n_classes, (batch_size // 2, ))
        latent_vector = torch.randn((batch_size // 2, latent_size))
        latent_vector = torch.cat([latent_vector, F.one_hot(labels, num_classes=n_classes)], dim=1)
        with torch.no_grad():
            generator.eval()
            x_generated = generator(latent_vector).detach()
            generator.train()

        x_train = torch.concat([x_real, x_generated])
        y_train = torch.concat([torch.full((batch_size // 2, 1), REAL_IMAGE),
                                torch.full((batch_size // 2, 1), FAKE_IMAGE)])

        # 1. train discriminator
        y_pred = discriminator(x_train)

        d_optimizer.zero_grad()
        d_loss = criterion(y_pred, y_train * 0.9)
        d_loss.backward()
        d_optimizer.step()

        # 2. train generator
        x_generated = generator(latent_vector)
        x_train = torch.concat([x_real, x_generated])
        y_pred = discriminator(x_train)

        g_optimizer.zero_grad()
        g_loss = criterion(y_pred, torch.full((batch_size, 1), REAL_IMAGE * 0.9))
        g_loss.backward()
        g_optimizer.step()

        # collect losses
        d_loss = d_loss.detach().numpy()
        g_loss = g_loss.detach().numpy()
        batch_losses.append((g_loss, d_loss))

        # collect accuracies
        with torch.no_grad():
            g_acc = g_accuracy(F.sigmoid(y_pred[batch_size // 2:]), torch.full((batch_size // 2, 1), REAL_IMAGE))
            d_acc = d_accuracy(F.sigmoid(y_pred), y_train)
            batch_accuracies.append((g_acc, d_acc))

        # print progress
        if step % progress_every == 0:
            print(f' ({epoch}/{step}) \tloss={d_loss + g_loss:0.3}, {g_loss=:0.3}, {d_loss=:0.3}')

    epoch_loss = np.mean(batch_losses, axis=0)
    losses.append(epoch_loss)
    accuracies.append(np.mean(batch_accuracies, axis=0))

    with torch.no_grad():
        print(f'>> {epoch=} '
              f'loss={epoch_loss.sum():0.3} '
              f'D(G(z))={F.sigmoid(y_pred[batch_size // 2:]).mean().item():0.3}')

    if do_inspections and epoch % inspections_every == 0:
        inspect_progress(epoch, x_train, discriminator)


# learning curves
gl, dl = zip(*losses)
plt.plot(gl, label='generator', alpha=0.5)
plt.plot(dl, label='discriminator', alpha=0.5)
plt.plot(np.asarray(gl) + np.asarray(dl), label='sum', alpha=0.5)
plt.title('Losses')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.savefig('losses.png')
plt.show()

# accuracies
ga, da = zip(*accuracies)
plt.plot(ga, label='generator', alpha=0.5)
plt.plot(da, label='discriminator', alpha=0.5)
plt.title('Accuracies')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.ylim((-0.1, 1.1))
plt.grid(axis='y')
plt.yticks([0, 0.5, 1])
plt.legend()
plt.tight_layout()
plt.savefig('accuracies.png')
plt.show()

# digit generation
with torch.no_grad():
    n_images = 100
    labels = torch.randint(0, n_classes, (n_images, ))
    latent_vector = torch.rand((n_images, latent_size))
    latent_vector = torch.cat([latent_vector, F.one_hot(labels, num_classes=n_classes)], dim=1)
    x_generated = generator(latent_vector)

    def normalize(x):
        """Go from [-1, 1] to [0, 1]."""
        return (x + 1) / 2

    grid = make_grid([normalize(x_generated[i]) for i in range(n_images)], nrow=10)
    plt.imshow(grid.permute(1, 2, 0), vmin=0, vmax=1)
    plt.axis('off')
    plt.title('Generated digits')
    plt.savefig('generated_digits.png')
    plt.tight_layout()
    plt.show()
