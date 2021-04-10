from dataclasses import dataclass
from time import time


@dataclass
class GANConfig:
    # Birads range
    birads_min: int = 2
    birads_max: int = 6

    # Whether to train conditional GAN
    conditional: bool = True
    split_birads_fours: bool = True

    # The number of condition labels for input into conditional GAN (i.e. 7 for BI-RADS 0 - 6)
    n_cond = birads_max + 1

    # l2 regularization in discriminator
    # value taken from here: https://machinelearningmastery.com/how-to-reduce-overfitting-in-deep-learning-with-weight-regularization/
    weight_decay: float = 0.0005

    # Number of workers for dataloader
    workers: int = 2

    # Batch size during training
    batch_size: int = 16

    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    image_size: int = 64

    # Whether to use least square loss
    use_lsgan_loss: bool = False

    # Leakiness for ReLUs
    leakiness: float = 0.2

    # Number of channels in the training images. For color images this is 3
    nc: int = 1

    # Size of z latent vector (i.e. size of generator input)
    nz: int = 100

    # Size of feature maps in generator
    ngf: int = 64

    # Size of feature maps in discriminator
    ndf: int = 64

    # Number of training epochs
    num_epochs: int = 60

    # Learning rate for optimizers
    lr: float = 0.0002

    # Beta1 hyperparam for Adam optimizers
    beta1: float = 0.5

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu: int = 1

    # The number of iterations between: i) prints ii) storage of results in tensorboard
    num_iterations_between_prints: int = 100

    # When plotting the discriminator accuracy, we need to set a threshold for its output in range [0,1]
    discriminator_clf_threshold: float = 0.5

    output_model_dir: str = f"model_checkpoints/training_{time()}/"

    def __post_init__(self):
        if self.conditional:
            self.nc = 2
        if self.split_birads_fours:
            self.birads_min = 1
            self.birads_max = 7
            self.n_cond = self.birads_max + 1
