import torch
import torch.nn as nn
import torch.nn.parallel
from gan_compare.training.networks.base_discriminator import BaseDiscriminator


class Discriminator(BaseDiscriminator):
    def __init__(
        self, ndf: int, nc: int, ngpu: int, leakiness: float = 0.2, bias: bool = False
    ):
        super(Discriminator, self).__init__(
            ndf=ndf,
            nc=nc,
            ngpu=ngpu,
            leakiness=leakiness,
            bias=bias,
        )
        self.num_embedding_input = 10
        self.num_embedding_dimensions = 50
        self.main = nn.Sequential(
            # input is (self.nc) x 64 x 64
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=self.bias),
            nn.LeakyReLU(leakiness, inplace=True),
            # state size. (self.ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=self.bias),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(leakiness, inplace=True),
            # state size. (self.ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=self.bias),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(leakiness, inplace=True),
            # state size. (self.ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=self.bias),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(leakiness, inplace=True),
            # state size. (self.ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=self.bias),
            nn.Sigmoid(),
        )
        self.embed_nn = nn.Sequential(
            # embedding layer
            nn.Embedding(
                num_embeddings=self.num_embedding_input,
                embedding_dim=self.num_embedding_dimensions,
            ),
            # target output dim of dense layer is (self.nc) x 64 x 64
            # input is dimension of the embedding layer output
            nn.Linear(in_features=self.num_embedding_dimensions, out_features=64 * 64),
            # nn.BatchNorm2d(2*64),
            nn.LeakyReLU(self.leakiness, inplace=True),
        )

    def forward(self, input_img, labels):
        # combining condition labels and input images via a new image channel
        # e.g. condition -> int -> embedding -> fcl -> feature map -> concat with image -> conv layers..
        # print(input_img.size())
        embedded_labels = self.embed_nn(labels)
        # print(embedded_labels.size())
        embedded_labels_as_image_channel = embedded_labels.view(-1, 1, 64, 64)
        # print(embedded_labels_as_image_channel.size())
        x = torch.cat([input_img, embedded_labels_as_image_channel], 1)
        return self.main(x)
