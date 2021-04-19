import torch
import torch.nn as nn
import torch.nn.parallel

from gan_compare.training.networks.base_discriminator import BaseDiscriminator


class Discriminator(BaseDiscriminator):
    def __init__(
            self, ndf: int, nc: int, ngpu: int, leakiness: float = 0.2, bias: bool = False, n_cond: int = 10,
            is_condition_categorical: bool = False,
    ):
        super(Discriminator, self).__init__(
            ndf=ndf,
            nc=nc,
            ngpu=ngpu,
            leakiness=leakiness,
            bias=bias,
        )
        # if is_condition_categorical is False, we model the condition as continous input to the network
        self.is_condition_categorical = is_condition_categorical

        # n_cond is only used if is_condition_categorical is True.
        self.num_embedding_input = n_cond

        # num_embedding_dimensions is only used if is_condition_categorical is True.
        self.num_embedding_dimensions = 50

        self.main = nn.Sequential(
            # input is (self.nc) x 64 x 64
            nn.Conv2d(self.nc, self.ndf, kernel_size=6, stride=2, padding=2, bias=self.bias),
            nn.LeakyReLU(leakiness, inplace=True),
            # state size. (self.ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, kernel_size=6, stride=2, padding=2, bias=self.bias),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(leakiness, inplace=True),
            # state size. (self.ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, kernel_size=6, stride=2, padding=2, bias=self.bias),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(leakiness, inplace=True),
            # state size. (self.ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, kernel_size=6, stride=2, padding=2, bias=self.bias),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(leakiness, inplace=True),
            # state size. (self.ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8, 1, kernel_size=6, stride=1, padding=1, bias=self.bias),
            nn.Sigmoid(),
        )
        self.embed_nn = nn.Sequential(
            # embedding layer
            nn.Embedding(
                num_embeddings=self.num_embedding_input,
                embedding_dim=self.num_embedding_dimensions,
            ),
            # target output dim of dense layer is batch_size x self.nc x 64 x 64
            # input is dimension of the embedding layer output
            nn.Linear(in_features=self.num_embedding_dimensions, out_features=64 * 64),
            # nn.BatchNorm1d(64*64),
            nn.LeakyReLU(self.leakiness, inplace=True),
        )
        self.embed_nn_only_linear = nn.Sequential(
            # target output dim of dense layer is batch_size x self.nc x 64 x 64
            # input is dimension of the conditional input
            nn.Linear(in_features=1, out_features=64 * 64),
            nn.BatchNorm1d(64 * 64),
            nn.LeakyReLU(self.leakiness, inplace=True),
        )

    def forward(self, input_img, labels):
        # combining condition labels and input images via a new image channel
        if self.is_condition_categorical:
            # e.g. condition -> int -> embedding -> fcl -> feature map -> concat with image -> conv layers..
            embedded_labels = self.embed_nn(labels)
        else:
            # e.g. condition -> float -> fcl -> concat with image -> conv layers..
            # If labels are continuous (not modelled as categorical), use floats instead of integers for labels.
            # Also adjust dimensions to (batch_size x 1) as needed for input into linear layer
            labels = labels.view(labels.size(0), -1).float()
            # Embed the labels using only a linear layer and passing them as float i.e. continuous conditional input
            embedded_labels = self.embed_nn_only_linear(labels)

        embedded_labels_as_image_channel = embedded_labels.view(-1, 1, 64, 64)
        x = torch.cat([input_img, embedded_labels_as_image_channel], 1)
        return self.main(x)
