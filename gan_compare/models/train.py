from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from gan_compare.models.config import * # TODO fix this ugly wildcart
from gan_compare.models.dataset import InbreastDataset 


if __name__ == "__main__":
    
    # Set random seed for reproducibility
    manualSeed = 999
    # manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)


    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    inbreast_dataset = InbreastDataset(metadata_path="metadata/metadata.json")
    dataloader = DataLoader(inbreast_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=workers)

    for i in range(0, 20):
        # Plot some training images
        real_batch = next(iter(dataloader))
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(np.concatenate((np.transpose(real_batch["image"][0]), np.transpose(real_batch["mask"][0]*255)), axis=0))
        plt.show()
