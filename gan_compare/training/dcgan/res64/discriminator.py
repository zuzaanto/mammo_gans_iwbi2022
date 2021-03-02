import torch
import torch.nn as nn
import torch.nn.parallel
from gan_compare.training.config import ndf, nc


class Discriminator(nn.Module):
    def __init__(self, ngpu, leakiness: float = 0.2, bias: bool = False):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.leakiness = leakiness
        self.bias = bias
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=self.bias),
            nn.LeakyReLU(leakiness, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=self.bias),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(leakiness, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=self.bias),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(leakiness, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=self.bias),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(leakiness, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=self.bias),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)