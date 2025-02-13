"""
Generator.
"""
from torch import nn
from torch.nn import functional as F


class Generator(nn.Module):

    def __init__(self, latent_size, n_classes, hidden_size, img_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.img_size = img_size
        self.fc = nn.Linear(latent_size + n_classes, hidden_size * img_size // 4 * img_size // 4)
        self.deconv1 = self.deconv_block(hidden_size, hidden_size // 2)
        self.deconv2 = self.deconv_block(hidden_size // 2, 1)

    def deconv_block(self, in_filters, out_filters):
        return nn.Sequential(
            nn.BatchNorm2d(in_filters),
            nn.ReLU(),
            nn.ConvTranspose2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], self.hidden_size, self.img_size // 4, self.img_size // 4)
        x = self.deconv1(x)
        x = self.deconv2(x)
        return F.sigmoid(x)
