import torch
import torch.nn as nn
import torch.nn.parallel
from gan_compare.training.networks.base_discriminator import BaseDiscriminator


class Discriminator(BaseDiscriminator):
    def __init__(self, ndf: int, nc: int, ngpu: int, leakiness: float = 0.2, bias: bool = False):
        super(Discriminator, self).__init__(
            ndf=ndf,
            nc=nc,
            ngpu=ngpu,
            leakiness=leakiness,
            bias=bias,
        )
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=self.bias),
            nn.LeakyReLU(self.leakiness, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=self.bias),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(self.leakiness, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=self.bias),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(self.leakiness, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=self.bias),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(self.leakiness, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=self.bias),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
    