from dataclasses import dataclass

from gan_compare.training.base_config import BaseConfig


@dataclass
class ClassifierConfig(BaseConfig):
    # Paths to train and validation metadata
    split_path: str = None

    train_shuffle_proportion: float = 0.4
    validation_shuffle_proportion: float = 0

    train_sampling_ratio: float = 0.5

    # Directory with synthetic patches
    synthetic_data_dir: str = None

    # Proportion of training artificial images
    gan_images_ratio: float = 0.5

    no_transforms: bool = False

    # Whether to use synthetic data at all
    use_synthetic: bool = True

    # Dropout rate
    dropout_rate: float = 0.3

    out_checkpoint_path: str = ""

    classes: str = "is_healthy" # one of ["is_benign", "is_healthy", "birads"]

    # Learning rate for optimizer
    lr: float = 0.0001  # Note: The CLF equivalent of the learning rates lr_g, lr_d1, lr_d2 in gan_config.py for GAN training.

    def __post_init__(self):
        super().__post_init__()
        self.out_checkpoint_path = f"{self.output_model_dir}/best_classifier.pt"
        assert self.classes in [
            "is_benign",
            "is_healthy",
            "birads",
        ], "Classifier currently supports either healthy vs unhealthy, or birads classification"  # TODO Add ACR classification

        if self.binary_classification:
            self.n_cond = 2
        elif self.split_birads_fours:
            self.birads_min = 1
            self.birads_max = 7
            self.n_cond = self.birads_max + 1

        assert (
            1 >= self.train_shuffle_proportion >= 0
        ), "Train shuffle proportion must be from <0,1> range"
        assert (
            1 >= self.validation_shuffle_proportion >= 0
        ), "Validation shuffle proportion must be from <0,1> range"
        if self.model_name == "swin_transformer":
            self.image_size = (
                224  # swin transformer currently only supports 224x224 images
            )
