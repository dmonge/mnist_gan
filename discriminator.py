"""
Discriminator.
"""
from torch import nn
from torch.nn import functional as F


REAL_IMAGE = 1.0
FAKE_IMAGE = 0.0


class Discriminator(nn.Module):

    def __init__(self, img_size, hidden_size):
        super().__init__()
        self.conv1 = self.conv_block(1, hidden_size // 2)
        self.conv2 = self.conv_block(hidden_size // 2, hidden_size)
        self.fc = nn.Linear(hidden_size * img_size // 4 * img_size // 4, 1)

    def conv_block(self, in_filters, out_filters):
        return nn.Sequential(
            nn.BatchNorm2d(in_filters),
            nn.ReLU(),
            nn.Conv2d(in_filters, out_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.sigmoid(x)
