from gan_compare.training.networks.classification.classifier_64 import Net as Net64
from gan_compare.training.networks.classification.classifier_128 import Net as Net128
import torch.nn as nn


class CNNNet(nn.Module):
    def __init__(self, num_classes: int, img_size: int) -> None:
        super(CNNNet, self).__init__()
        if img_size == 128:
            self.__dict__ = Net128(num_labels=num_classes).__dict__.copy()
        elif img_size == 64:
            self.__dict__ = Net64(num_labels=num_classes).__dict__.copy()
        else:
            raise ValueError(f"Image size {img_size} not supported. Quitting.")
        self.img_size = img_size
        self.num_classes = num_classes

    def forward(self, s):
        if self.img_size == 128:
            return Net128(num_labels=self.num_classes).forward(s)
        elif self.img_size == 64:
            return Net64(num_labels=self.num_classes).forward(s)
        else:
            raise ValueError(f"Image size {self.img_size} not supported. Quitting.")
