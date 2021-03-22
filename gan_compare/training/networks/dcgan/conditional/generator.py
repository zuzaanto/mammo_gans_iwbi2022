import torch
import torch.nn as nn
import torch.nn.parallel
from gan_compare.training.networks.base_generator import BaseGenerator


class Generator(BaseGenerator):
    def __init__(self, nz: int, ngf: int, nc: int, ngpu: int, leakiness: float = 0.2, bias: bool = False):
        super(Generator, self).__init__(
            nz=nz,
            ngf=ngf,
            nc=nc,
            ngpu=ngpu,
            leakiness=leakiness,
            bias=bias,
        )
        self.num_embedding_input = 10
        self.num_embedding_dimensions = 50
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.nz * 2, self.ngf * 8, 4, 1, 0, bias=self.bias),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=self.bias),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=self.bias),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=self.bias),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(self.ngf, 1, 4, 2, 1, bias=self.bias),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
        self.embed_nn = nn.Sequential(
            # embedding layer
            nn.Embedding(num_embeddings=self.num_embedding_input, embedding_dim=self.num_embedding_dimensions),
            # target output dim of dense layer is: nz x 1 x 1
            # input is dimension of the embedding layer output
            nn.Linear(in_features=self.num_embedding_dimensions, out_features=self.nz),
            # nn.BatchNorm2d(10*10),
            nn.LeakyReLU(self.leakiness, inplace=True)
        )

    def forward(self, rand_input, labels):
        
        # combining condition labels and input images via a new image channel
        # e.g. condition -> int -> embedding -> fcl -> feature map -> concat with image -> conv layers..
        # print(rand_input.size())
        # print(labels.size())
        embedded_labels = self.embed_nn(labels)
        # print(embedded_labels.size())
        embedded_labels_with_random_noise_dim = embedded_labels.view(-1, 100, 1, 1)
        # print(embedded_labels_with_random_noise_dim.size())
        x = torch.cat([rand_input, embedded_labels_with_random_noise_dim], 1)
        return self.main(x)
