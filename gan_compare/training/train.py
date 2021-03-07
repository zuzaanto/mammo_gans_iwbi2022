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
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
from gan_compare.training.config import * # TODO fix this ugly wildcart
from gan_compare.training.dataset import InbreastDataset 
from gan_compare.training.dcgan.utils import weights_init
from gan_compare.data_utils.utils import interval_mapping


def parse_args()-> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, required=True, help="Model name: supported: dcgan and lsgan"
    )
    parser.add_argument(
        "--save_dataset", action="store_true", help="Whether to save the dataset samples."
    )
    parser.add_argument(
        "--out_dataset_path", type=str, default="visualisation/inbreast_dataset/", help="Directory to save the dataset samples in."
    )
    args = parser.parse_args()
    return args
    

if __name__ == "__main__":
    args = parse_args()
    # Set random seed for reproducibility
    manualSeed = 999
    # manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)


    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    inbreast_dataset = InbreastDataset(metadata_path="metadata/metadata.json", final_shape=(image_size, image_size))
    dataloader = DataLoader(inbreast_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=workers)
    if args.save_dataset:
        output_dataset_dir = Path(args.out_dataset_path)
        if not output_dataset_dir.exists():
            os.makedirs(output_dataset_dir.resolve())
        for i in range(len(inbreast_dataset)):
            print(inbreast_dataset[i])
                # Plot some training images
            cv2.imwrite(str(output_dataset_dir / f"{i}.png"), inbreast_dataset.__getitem__(i, to_save=True))
    
    # Create the Discriminator
    if args.model_name =="dcgan":
        if image_size == 64:
            from gan_compare.training.dcgan.res64.discriminator import Discriminator
            from gan_compare.training.dcgan.res64.generator import Generator

            netG = Generator(ngpu).to(device)
            netD = Discriminator(ngpu).to(device)
        elif image_size == 128:
            from gan_compare.training.dcgan.res128.discriminator import Discriminator
            from gan_compare.training.dcgan.res128.generator import Generator

            netD = Discriminator(ngpu, leakiness=leakiness, bias=False).to(device)
            netG = Generator(ngpu).to(device)
        else:
            raise ValueError("Unsupported image size. Supported sizes are 128 and 64.")
    elif args.model_name == "lsgan":
        # only 64x64 image resolution will be supported
        assert image_size == 64, "Wrong image size for LSGAN, change it to 64x64 before proceeding."
        
        from gan_compare.training.lsgan.discriminator import Discriminator
        from gan_compare.training.lsgan.generator import Generator

        netG = Generator().to(device)
        netD = Discriminator().to(device)
    else:
        raise ValueError(f"Unknown model name: {args.model_name}")


    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netG.apply(weights_init)

    # Print the model
    print(netG)
    
    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))
        
    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)

    # Print the model
    print(netD)
        
    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(image_size, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay=weight_decay)
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            if args.model_name != "lsgan" and not use_lsgan_loss:
                errD_real = criterion(output, label)
            else:
                errD_real = 0.5 * torch.mean((output-label)**2)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            if args.model_name != "lsgan" and not use_lsgan_loss:
                errD_fake = criterion(output, label)
            else:
                errD_fake = 0.5 * torch.mean((output-label)**2)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            if args.model_name != "lsgan" and not use_lsgan_loss:
                errG = criterion(output, label)
            else:
                errG = 0.5 * torch.mean((output-label)**2)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()
            
            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            
            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            
            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(fake.numpy())
                # img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            iters += 1
    output_model_dir = Path(output_model_dir)
    out_path = output_model_dir / f"{args.model_name}.pt"
    if not output_model_dir.exists():
        os.makedirs(output_model_dir.resolve())
    torch.save({
        "discriminator": netD.state_dict(), 
        "generator": netG.state_dict(), 
        "optim_discriminator": optimizerD.state_dict(), 
        "optim_generator": optimizerG.state_dict(),
    }, out_path)
    print(f"Saved model to {out_path.resolve()}")
    for i, image_batch in enumerate(img_list):
        for j, img_ in enumerate(image_batch):
            img_path = output_model_dir / f"{args.model_name}_{i}_{j}.png"
            img_ = interval_mapping(img_.transpose(1, 2, 0), -1., 1., 0, 255)
            img_ = img_.astype('uint8')
            # print(img_)
            # print(img_.shape)
            cv2.imwrite(str(img_path.resolve()), img_)
        