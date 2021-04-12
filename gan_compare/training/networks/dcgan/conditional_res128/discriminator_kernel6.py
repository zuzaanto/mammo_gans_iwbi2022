import torch
import torch.nn as nn
import torch.nn.parallel
from gan_compare.training.networks.base_discriminator import BaseDiscriminator


class Discriminator(BaseDiscriminator):
    def __init__(
        self, ndf: int, nc: int, ngpu: int, leakiness: float = 0.2, bias: bool = False, n_cond: int = 6
    ):
        super(Discriminator, self).__init__(
            ndf=ndf,
            nc=nc,
            ngpu=ngpu,
            leakiness=leakiness,
            bias=bias,
        )
        self.num_embedding_input = n_cond  # number of possible conditional values
        self.num_embedding_dimensions = 50  # standard would probably be dim(z). Using same value in D and G.
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(in_channels=self.nc, out_channels=self.ndf, kernel_size=6, stride=2, padding=2, bias=self.bias),
            nn.LeakyReLU(self.leakiness, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(self.ndf, self.ndf * 2, kernel_size=6, stride=2, padding=2, bias=self.bias),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(self.leakiness, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(
                self.ndf * 2, self.ndf * 4, kernel_size=6, stride=2, padding=2, bias=self.bias
            ),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(self.leakiness, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(
                self.ndf * 4, self.ndf * 8, kernel_size=6, stride=2, padding=2, bias=self.bias
            ),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(self.leakiness, inplace=True),
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(
                self.ndf * 8, self.ndf * 16, kernel_size=6, stride=2, padding=2, bias=self.bias
            ),
            nn.BatchNorm2d(self.ndf * 16),
            nn.LeakyReLU(self.leakiness, inplace=True),
            # state size. (ndf*16) x 4 x 4
            nn.Conv2d(self.ndf * 16, 1, kernel_size=6, stride=1, padding=1, bias=self.bias),
            nn.Sigmoid()
            # state size. 1
        )
        self.embed_nn = nn.Sequential(
            # embedding layer
            nn.Embedding(
                num_embeddings=self.num_embedding_input,
                embedding_dim=self.num_embedding_dimensions,
            ),
            # target output dim of dense layer is batch_size x self.nc x 128 x 128
            # input is dimension of the embedding layer output
            nn.Linear(in_features=self.num_embedding_dimensions, out_features=128 * 128),
            # nn.BatchNorm2d(2*128),
            nn.LeakyReLU(self.leakiness, inplace=True),
        )

    def forward(self, input_img, labels):
        # combining condition labels and input images via a new image channel
        # e.g. condition -> int -> embedding -> fcl -> feature map -> concat with image -> conv layers..
        # print(input_img.size())
        embedded_labels = self.embed_nn(labels)
        # print(embedded_labels.size())
        embedded_labels_as_image_channel = embedded_labels.view(-1, 1, 128, 128)
        # print(embedded_labels_as_image_channel.size())
        x = torch.cat([input_img, embedded_labels_as_image_channel], 1)
        return self.main(x)
