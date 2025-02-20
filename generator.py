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
        self.input = self.input_block(latent_size, n_classes, hidden_size, img_size)
        self.deconv1 = self.deconv_block(hidden_size, hidden_size // 2)
        self.deconv2 = self.deconv_block(hidden_size // 2, hidden_size // 4)
        self.deconv3 = nn.ConvTranspose2d(hidden_size // 4, 1, kernel_size=4, stride=2, padding=1)
        self.upsample = nn.Upsample(size=(img_size, img_size), mode='bilinear')

    def input_block(self, latent_size, n_classes, hidden_size, img_size):
        in_features = latent_size + n_classes
        out_features = hidden_size * (img_size // 8) * (img_size // 8)
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
        )

    def deconv_block(self, in_filters, out_filters):
        return nn.Sequential(
            nn.ConvTranspose2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_filters),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.input(x)
        x = x.view(x.shape[0], self.hidden_size, self.img_size // 8, self.img_size // 8)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.upsample(x)
        return F.tanh(x)
