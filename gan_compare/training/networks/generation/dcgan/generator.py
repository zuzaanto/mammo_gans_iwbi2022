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

        if self.is_condition_categorical:
            self.embed_nn = nn.Sequential(
                # e.g. condition -> int -> embedding -> fcl -> feature map -> concat with image -> conv layers..
                # embedding layer
                nn.Embedding(
                    num_embeddings=self.num_embedding_input,
                    embedding_dim=self.num_embedding_dimensions,
                ),
                # target output dim of dense layer is batch_size x self.nz x 1 x 1
                # input is dimension of the embedding layer output
                nn.Linear(
                    in_features=self.num_embedding_dimensions, out_features=self.nz
                ),
                nn.LeakyReLU(self.leakiness, inplace=True),
            )
        else:
            self.embed_nn = nn.Sequential(
                # target output dim of dense layer is: nz x 1 x 1
                # input is dimension of the numbers of embedding
                nn.Linear(in_features=1, out_features=self.nz),
                # TODO Ablation: How does BatchNorm1d affect the conditional model performance?
                # Outcommenting batchnorm here to avoid value errors (Expected more than 1 value per channel..) when batch_size is one.
                # nn.BatchNorm1d(self.nz),
                nn.LeakyReLU(self.leakiness, inplace=True),
            )

    def forward(self, x, conditions=None):
        if self.conditional:
            # combining condition labels and input images via a new image channel
            if not self.is_condition_categorical:
                # If labels are continuous (not modelled as categorical), use floats instead of integers for labels.
                # Also adjust dimensions to (batch_size x 1) as needed for input into linear layer
                # labels should already be of type float, no change expected in .float() conversion (it is only a safety check)

                # Just for testing:
                # conditions *= 0
                # conditions += 1

                conditions = conditions.view(conditions.size(0), -1).float()
            embedded_conditions = self.embed_nn(conditions)
            embedded_conditions_with_random_noise_dim = embedded_conditions.view(
                -1, self.nz, 1, 1
            )
            x = torch.cat([x, embedded_conditions_with_random_noise_dim], 1)
        return self.main(x)
