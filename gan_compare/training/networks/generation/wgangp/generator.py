import torch
import torch.nn as nn
import torch.nn.parallel

from gan_compare.training.networks.generation.base_generator import BaseGenerator


class Generator(BaseGenerator):
    def __init__(
        self,
        nz: int,
        ngf: int,
        nc: int,
        ngpu: int,
        image_size: int,
        conditional: bool,
        leakiness: float,
        bias: bool = False,
        n_cond: int = 10,
        is_condition_categorical: bool = False,
        num_embedding_dimensions: int = 50,
        **kwargs,
    ):
        super(Generator, self).__init__(
            nz=nz,
            ngf=ngf,
            nc=nc,
            ngpu=ngpu,
            leakiness=leakiness,
            bias=bias,
        )

        # Only non-conditional training for now.
        assert (
            conditional == False
        ), f"WGAN-GP only supports non-conditional training at the moment. Please set 'conditional' to False."

        # if is_condition_categorical is False, we model the condition as continous input to the network
        self.is_condition_categorical = is_condition_categorical

        # n_cond is only used if is_condition_categorical is True.
        self.num_embedding_input = n_cond

        # num_embedding_dimensions is only used if is_condition_categorical is True.
        # num_embedding_dimensions standard would be dim(z), but atm we have a nn.Linear after
        # nn.Embedding that upscales the dimension to self.nz. Using same value of num_embedding_dims in D and G.
        self.num_embedding_dimensions = num_embedding_dimensions

        # whether the is a conditional input into the GAN i.e. cGAN
        self.conditional: bool = conditional
        if conditional:
            # ingesting condition as additional channel into G (and D).
            self.cond_channel = 1
        else:
            self.cond_channel = 0

        # The image size (supported params should be 128 or 64)
        self.image_size = image_size

        # TODO Check if architecture works better with the initial linear layer in DCGAN added by the WGAN-GP authors:
        # https://github.com/igul222/improved_wgan_training/blob/fa66c574a54c4916d27c55441d33753dcc78f6bc/gan_64x64.py#L242

        if self.image_size == 224:
            self.first_layers = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(
                    self.nz * (1 + self.cond_channel),
                    self.ngf * 32,
                    4,
                    1,
                    0,
                    bias=self.bias,
                ),
                nn.BatchNorm2d(self.ngf * 32),
                nn.ReLU(True),
                # state size. (ngf*32) x 4 x 4
                nn.ConvTranspose2d(
                    self.ngf * 32, self.ngf * 16, 4, 1, 0, bias=self.bias
                ),
                nn.BatchNorm2d(self.ngf * 16),
                nn.ReLU(True),
                # state size. (ngf*16) x 7 x 7
                nn.ConvTranspose2d(
                    self.ngf * 16, self.ngf * 8, 4, 2, 1, bias=self.bias
                ),
                nn.BatchNorm2d(self.ngf * 8),
                nn.ReLU(True),
            )
        elif self.image_size == 128:
            self.first_layers = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(
                    self.nz * (1 + self.cond_channel),
                    self.ngf * 16,
                    4,
                    1,
                    0,
                    bias=self.bias,
                ),
                nn.BatchNorm2d(self.ngf * 16),
                nn.ReLU(True),
                # state size. (ngf*16) x 4 x 4
                nn.ConvTranspose2d(
                    self.ngf * 16, self.ngf * 8, 4, 2, 1, bias=self.bias
                ),
                nn.BatchNorm2d(self.ngf * 8),
                nn.ReLU(True),
            )
        elif self.image_size == 64:
            self.first_layers = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(
                    self.nz * (1 + self.cond_channel),
                    self.ngf * 8,
                    4,
                    1,
                    0,
                    bias=self.bias,
                ),
                nn.BatchNorm2d(self.ngf * 8),
                nn.ReLU(True),
            )
        else:
            raise ValueError(
                f"Allowed image sizes are 224, 128 and 64. You provided {self.image_size}. Please adjust."
            )

        self.main = nn.Sequential(
            *self.first_layers.children(),
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
            nn.ConvTranspose2d(
                in_channels=self.ngf,
                out_channels=self.nc - self.cond_channel,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=self.bias,
            ),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )

    def forward(self, x, conditions=None):
        return self.main(x)
