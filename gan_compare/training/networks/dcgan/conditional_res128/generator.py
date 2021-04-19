import torch
import torch.nn as nn
import torch.nn.parallel

from gan_compare.training.networks.base_generator import BaseGenerator


class Generator(BaseGenerator):
    def __init__(
        self,
        nz: int,
        ngf: int,
        nc: int,
        ngpu: int,
        leakiness: float = 0.2,
        bias: bool = False,
        n_cond: int = 10,
        is_condition_categorical: bool = False,
    ):
        super(Generator, self).__init__(
            nz=nz,
            ngf=ngf,
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
        # num_embedding_dimensions standard would be dim(z), but atm we have a nn.Linear after
        # nn.Embedding that upscales the dimension to self.nz. Using same value of num_embedding_dims in D and G.
        self.num_embedding_dimensions = 50

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.nz * self.nc, self.ngf * 16, 4, 1, 0, bias=self.bias),
            nn.BatchNorm2d(self.ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*16) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 16, self.ngf * 8, 4, 2, 1, bias=self.bias),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=self.bias),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=self.bias),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=self.bias),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            # Note that out_channels=1 instead of out_channels=self.nc.
            # This is due to conditional input channel of our grayscale images
            nn.ConvTranspose2d(in_channels=self.ngf, out_channels=1, kernel_size=4, stride=2, padding=1,
                               bias=self.bias),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )
        self.embed_nn = nn.Sequential(
            # embedding layer
            nn.Embedding(
                num_embeddings=self.num_embedding_input,
                embedding_dim=self.num_embedding_dimensions,
            ),
            # target output dim of dense layer is: nz x 1 x 1
            # input is dimension of the embedding layer output
            nn.Linear(in_features=self.num_embedding_dimensions, out_features=self.nz),
            # nn.BatchNorm1d(10*10),
            nn.LeakyReLU(self.leakiness, inplace=True),
        )
        self.embed_nn_only_linear = nn.Sequential(
            # target output dim of dense layer is: nz x 1 x 1
            # input is dimension of the numbers of embedding
            nn.Linear(in_features=1, out_features=self.nz),
            # TODO Ablation: Test impact of BatchNorm1d here - how does it affect the conditional model performance?
            nn.BatchNorm1d(self.nz),
            nn.LeakyReLU(self.leakiness, inplace=True),
        )

    def forward(self, rand_input, labels):

        # combining condition labels and input images via a new image channel
        if self.is_condition_categorical:
            # e.g. condition -> int -> embedding -> fcl -> feature map -> concat with image -> conv layers..
            embedded_labels = self.embed_nn(labels)
        else:
            # e.g. condition -> float -> fcl -> concat with image -> conv layers..
            # If labels are continuous (not modelled as categorical), use floats instead of integers for labels.
            # Also adjust dimensions to (16 x 1) as needed for input into linear layer
            labels = labels.view(labels.size(0), -1).float()

            # Embed the labels using only a linear layer and passing them as float i.e. continuous conditional input
            embedded_labels = self.embed_nn_only_linear(labels)

        embedded_labels_with_random_noise_dim = embedded_labels.view(-1, 100, 1, 1)
        x = torch.cat([rand_input, embedded_labels_with_random_noise_dim], 1)
        return self.main(x)