"""
Discriminator.
"""
import torch
from torch import nn


REAL_IMAGE = 1.0
FAKE_IMAGE = 0.0


class Discriminator(nn.Module):

    def __init__(self, img_size, hidden_size, mbd_in_features=100, mbd_out_features=50, mbd_kernel_dims=5):
        super().__init__()
        self.conv1 = self.conv_block(1, hidden_size // 2)
        self.conv2 = self.conv_block(hidden_size // 2, hidden_size)
        self.fc1 = nn.utils.spectral_norm(nn.Linear(hidden_size * img_size // 4 * img_size // 4, mbd_in_features))
        self.disc = MinibatchDiscrimination(100, mbd_out_features, mbd_kernel_dims)
        self.fc2 = nn.Linear(mbd_in_features + mbd_out_features, 1)

    def conv_block(self, in_filters, out_filters):
        return nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(in_filters, out_filters, kernel_size=3, padding=1)),
            nn.BatchNorm2d(out_filters),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = x + 0.1 + torch.randn_like(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.disc(x)
        x = self.fc2(x)
        return x


class MinibatchDiscrimination(nn.Module):

    def __init__(self, in_features, out_features, kernel_dims):
        super().__init__()
        self.T = nn.Parameter(torch.randn(in_features, out_features, kernel_dims))

    def forward(self, x):
        in_features, out_features, kernel_dims = self.T.shape
        m = x.matmul(self.T.view(in_features, -1))
        m = m.view(x.shape[0], out_features, kernel_dims)
        diff = m.unsqueeze(0) - m.unsqueeze(1)
        norm = torch.sum(torch.abs(diff), dim=3)
        norm = torch.exp(-norm)
        features = torch.sum(norm, dim=1)
        return torch.cat([x, features], dim=1)
