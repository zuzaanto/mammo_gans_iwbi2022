from dataclasses import dataclass
from typing import Optional


@dataclass
class ClassifierConfig:
    # Paths to train and validation metadata
    train_metadata_path: str
    validation_metadata_path: str
    test_metadata_path: str
    
    # Path to synthetic metadata used for data augmentation
    synthetic_metadata_path: str
    train_shuffle_proportion: float
    validation_shuffle_proportion: float
    
    # Proportion of training artificial images
    gan_images_ratio: float = 0.4

    # Birads range
    birads_min: int = 2
    birads_max: int = 6

    split_birads_fours: bool = True

    # The number of condition labels for input into conditional GAN (i.e. 7 for BI-RADS 0 - 6)
    n_cond = birads_max + 1

    # Number of workers for dataloader
    workers: int = 2

    # Batch size during training
    batch_size: int = 8

    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    image_size: int = 64

    # Dropout rate
    dropout_rate: float = 0.3

    # Number of training epochs
    num_epochs: int = 60

    # Learning rate for optimizers
    lr: float = 0.0002

    out_checkpoint_path: str = "model_checkpoints//classifier/classifier.pt"

    def __post_init__(self):
        if self.split_birads_fours:
            self.birads_min = 1
            self.birads_max = 7
            self.n_cond = self.birads_max + 1
        assert 1 >= self.train_shuffle_proportion >= 0, "Train shuffle proportion must be from <0,1> range"
        assert 1 >= self.validation_shuffle_proportion >= 0, "Validation shuffle proportion must be from <0,1> range"
