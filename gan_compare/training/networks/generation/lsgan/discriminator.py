"""
Implementation taken from https://github.com/meliketoy/LSGAN.pytorch
"""
import torch.nn as nn

from gan_compare.training.networks.generation.base_discriminator import BaseDiscriminator


class Discriminator(BaseDiscriminator):
    def __init__(
        self, ndf: int, nc: int, ngpu: int, leakiness: float, bias: bool = False
    ):
        super(Discriminator, self).__init__(
            ndf=ndf,
            nc=nc,
            ngpu=ngpu,
            leakiness=leakiness,
            bias=bias,
        )
        # input : (batch * nc * image width * image height)
        # Discriminator will be consisted with a series of convolution networks

        self.layer1 = nn.Sequential(
            # Input size : input image with dimension (nc)*64*64
            # Output size: output feature vector with (ndf)*32*32
            nn.Conv2d(
                in_channels=nc,
                out_channels=ndf,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer2 = nn.Sequential(
            # Input size : input feature vector with (ndf)*32*32
            # Output size: output feature vector with (ndf*2)*16*16
            nn.Conv2d(
                in_channels=ndf,
                out_channels=ndf * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer3 = nn.Sequential(
            # Input size : input feature vector with (ndf*2)*16*16
            # Output size: output feature vector with (ndf*4)*8*8
            nn.Conv2d(
                in_channels=ndf * 2,
                out_channels=ndf * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer4 = nn.Sequential(
            # Input size : input feature vector with (ndf*4)*8*8
            # Output size: output feature vector with (ndf*8)*4*4
            nn.Conv2d(
                in_channels=ndf * 4,
                out_channels=ndf * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer5 = nn.Sequential(
            # Input size : input feature vector with (ndf*8)*4*4
            # Output size: output probability of fake/real image
            nn.Conv2d(
                in_channels=ndf * 8,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            # nn.Sigmoid() -- Replaced with Least Square Loss
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)

        return out.view(-1, 1)


# if __name__ == "__main__":
#     net = Discriminator(
#         nc = 3,
#         ndf = 64
#     )
#     print "Input(=image) : ",
#     print(torch.randn(128,3,64,64).size())
#     y = net(Variable(torch.randn(128,3,64,64))) # Input should be a 4D tensor
#     print "Output(batchsize, channels, width, height) : ",
#     print(y.size())
