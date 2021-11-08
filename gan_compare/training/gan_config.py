from dataclasses import dataclass, field
from time import time
from typing import List
from gan_compare.constants import DATASET_DICT


@dataclass
class GANConfig:
    # l2 regularization in discriminator value taken from here:
    # https://machinelearningmastery.com/how-to-reduce-overfitting-in-deep-learning-with-weight-regularization/
    # "weight decay often encourage some misclassification if the coefficient on the regularizer is set high enough"
    # - https://arxiv.org/pdf/1701.00160.pdf
    weight_decay: float = 0 #5e-06  # 0.000005

    # Number of workers for dataloader
    workers: int = 2

    # Batch size during training
    batch_size: int = 16

    # Spatial size of training images. All images will be resized to this
    # size using a transformer.
    image_size: int = 128

    # Whether to use least square loss
    use_lsgan_loss: bool = False

    # Whether to switch the loss function (i.e. from ls to bce) on each epoch.
    switch_loss_each_epoch: bool = False

    # Whether the discriminator kernel size should be changed from 4 to 6. D's kernel size of 6 does away with the
    # symmetry between discriminator and generator kernel size (=4). This symmetry can cause checkerboard effects as
    # it introduces blind spots for the discriminator as described in https://arxiv.org/pdf/1909.02062.pdf
    kernel_size: int = 6

    # Reduce the overconfidence of predictions of the discriminator for positive labels by replacing only the label
    # for real images(=1) with a value smaller than 1 (described in https://arxiv.org/pdf/1701.00160.pdf) or with a
    # value randomly drawn from an interval surrounding 1, e.g. (0.7,1.2) (described in
    # https://github.com/soumith/ganhacks#6-use-soft-and-noisy-labels).
    use_one_sided_label_smoothing: bool = True
    # Define the one-sided label smoothing interval for positive labels (real images) for D.
    label_smoothing_start: float = 0.95
    label_smoothing_end: float = 1.0

    # Leakiness for ReLUs
    leakiness: float = 0.1

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

    # Specify whether ROIs of masses should be included into GAN training
    is_trained_on_masses: bool = True

    # Specify whether ROIs of calcifications should be included into GAN training
    is_trained_on_calcifications: bool = False

    # Specify whether other ROI types (e.g. Assymetry, Distortion, etc) should be included into GAN training
    is_trained_on_other_roi_types: bool = False

    # Specify whether basic data augmentation methods should be applied to the GAN training data.
    is_training_data_augmented: bool = True

    output_model_dir: str = f"model_checkpoints/training_{time()}/"

    dataset_names: List[str] = field(default_factory=list)

    ########## Start: Variables related to condition ###########
    # Whether to train conditional GAN
    conditional: bool = True

    # determines if we model the condition in the nn as either continuous (False) or discrete/categorical (True)
    is_condition_categorical: bool = False

    # Specifiy whether birads condition is modeled as binary e.g., benign/malignant with birads 1-3 = 0, 4-6 = 1
    is_condition_binary: bool = True

    # We can condition on different variables such as breast density or birads status of lesion. Default = "density"
    conditioned_on: str = "density"

    # the minimum possible value that the condition can have
    condition_min: int = 1

    # the maximum possible value that the condition can have
    condition_max: int = 4

    # To have more variation in continuous conditional variables, we can add to them some random noise drawn
    # from [0,1] multiplied by the added_noise_term. The hope is that this reduces mode collapse.
    added_noise_term: float = 0.5

    # 4a 4b 4c of birads are splitted into integers
    split_birads_fours: bool = True

    # The number of condition labels for input into conditional GAN (i.e. 7 for BI-RADS 0 - 6)
    n_cond: int = condition_max + 1

    # The dimension of embedding tensor in torch.nn.embedding layers in G and D in categorical c-GAN setting.
    num_embedding_dimensions: int = 50

    ########## End: Variables related to condition ###########

    def __post_init__(self):
        if self.conditional:
            self.nc = 2
            if self.is_condition_binary:
                self.condition_min = 0
                self.condition_max = 1
            elif self.conditioned_on == "density":
                # Density range (called "ACR" in InBreast)
                self.condition_min = 1
                self.condition_max = 4
            elif self.conditioned_on == "birads":
                if self.split_birads_fours:
                    self.birads_min = 1
                    self.birads_max = 7
                else:
                    self.condition_min = 2
                    self.condition_max = 6
            self.n_cond = self.condition_max + 1
        assert all(dataset_name in ["bcdr", "inbreast"] for dataset_name in self.dataset_names)
