import torch
import torch.nn as nn
import torch.nn.parallel

from gan_compare.training.networks.base_discriminator import BaseDiscriminator


class Discriminator(BaseDiscriminator):
    def __init__(
            self, ndf: int, nc: int, ngpu: int, image_size: int, conditional: bool, leakiness: float = 0.2,
            bias: bool = False, n_cond: int = 10, is_condition_categorical: bool = False,
            num_embedding_dimensions: int = 50, kernel_size: int = 6,
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
        self.num_embedding_dimensions = num_embedding_dimensions

        # whether the is a conditional input into the GAN i.e. cGAN
        self.conditional: bool = conditional

        # the kernel size (supported params should be 6 or 4)
        self.kernel_size = kernel_size

        # The image size (supported params should be 128 or 64)
        self.image_size = image_size

        stride = 2
        padding = 2
        if self.kernel_size == 4:
            padding = 1
        elif kernel_size != 6:
            raise ValueError(f"Allowed kernel sizes are 6 and 4. You provided {self.kernel_size}. Please adjust.")

        if self.image_size == 128:
            self.ndf_input_main = self.ndf * 2
            self.first_layers = nn.Sequential(
                # input is (nc) x 128 x 128
                nn.Conv2d(in_channels=self.nc, out_channels=self.ndf, kernel_size=self.kernel_size, stride=stride,
                          padding=padding,
                          bias=self.bias),
                nn.LeakyReLU(self.leakiness, inplace=True),
                # state size. (ndf) x 64 x 64
                nn.Conv2d(self.ndf, self.ndf * 2, kernel_size=self.kernel_size, stride=stride, padding=padding,
                          bias=self.bias),
                nn.BatchNorm2d(self.ndf * 2),
                nn.LeakyReLU(self.leakiness, inplace=True),
            )
        elif self.image_size == 64:
            self.ndf_input_main = self.ndf
            self.first_layers = nn.Sequential(
                # input is (self.nc) x 64 x 64
                nn.Conv2d(self.nc, self.ndf, kernel_size=self.kernel_size, stride=stride, padding=padding,
                          bias=self.bias),
                nn.LeakyReLU(leakiness, inplace=True),
            )
        else:
            raise ValueError(f"Allowed image sizes are 128 and 64. You provided {self.image_size}. Please adjust.")

        self.main = nn.Sequential(
            *self.first_layers.children(),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(
                self.ndf_input_main, self.ndf_input_main * 2, kernel_size=self.kernel_size, stride=stride,
                padding=padding, bias=self.bias
            ),
            nn.BatchNorm2d(self.ndf_input_main * 2),
            nn.LeakyReLU(self.leakiness, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(
                self.ndf_input_main * 2, self.ndf_input_main * 4, kernel_size=self.kernel_size, stride=stride,
                padding=padding, bias=self.bias
            ),
            nn.BatchNorm2d(self.ndf_input_main * 4),
            nn.LeakyReLU(self.leakiness, inplace=True),
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(
                self.ndf_input_main * 4, self.ndf_input_main * 8, kernel_size=self.kernel_size, stride=stride,
                padding=padding, bias=self.bias
            ),
            nn.BatchNorm2d(self.ndf_input_main * 8),
            nn.LeakyReLU(self.leakiness, inplace=True),
            # state size. (ndf*16) x 4 x 4
            nn.Conv2d(self.ndf_input_main * 8, 1, kernel_size=self.kernel_size, stride=stride - 1, padding=padding - 1,
                      bias=self.bias),
            nn.Sigmoid(),
            # state size. 1
        )

        if self.is_condition_categorical:
            self.embed_nn = nn.Sequential(
                # e.g. condition -> int -> embedding -> fcl -> feature map -> concat with image -> conv layers..
                # embedding layer
                nn.Embedding(
                    num_embeddings=self.num_embedding_input,
                    embedding_dim=self.num_embedding_dimensions,
                ),
                # target output dim of dense layer is batch_size x self.nc x 128 x 128
                # input is dimension of the embedding layer output
                nn.Linear(in_features=self.num_embedding_dimensions, out_features=self.image_size * self.image_size),
                # nn.BatchNorm1d(self.image_size*self.image_size),
                nn.LeakyReLU(self.leakiness, inplace=True),
            )
        else:
            self.embed_nn = nn.Sequential(
                # e.g. condition -> float -> fcl -> concat with image -> conv layers..
                # Embed the labels using only a linear layer and passing them as float i.e. continuous conditional input
                # target output dim of dense layer is batch_size x self.nc x 128 x 128
                # input is dimension of the conditional input
                nn.Linear(in_features=1, out_features=self.image_size * self.image_size),
                nn.BatchNorm1d(self.image_size * self.image_size),
                nn.LeakyReLU(self.leakiness, inplace=True),
            )

    def forward(self, x, conditions=None):
        if self.conditional:
            # combining condition labels and input images via a new image channel
            if not self.is_condition_categorical:
                # If labels are continuous (not modelled as categorical), use floats instead of integers for labels.
                # Also adjust dimensions to (batch_size x 1) as needed for input into linear layer
                # labels should already be of type float, no change expected in .float() conversion (it is only a safety check)
                conditions = conditions.view(conditions.size(0), -1).float()
            embedded_conditions = self.embed_nn(conditions)
            embedded_conditions_as_image_channel = embedded_conditions.view(-1, 1, self.image_size, self.image_size)
            x = torch.cat([x, embedded_conditions_as_image_channel], 1)
        return self.main(x)
