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
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
from typing import Union
from gan_compare.training.gan_config import GANConfig
from gan_compare.training.networks.dcgan.utils import weights_init
from gan_compare.data_utils.utils import interval_mapping
from gan_compare.training.io import save_yaml


class GANModel:
    def __init__(
        self,
        model_name: str, 
        config: GANConfig, 
        dataloader: DataLoader,
        out_dataset_path: Union[str, Path] = "visualisation/inbreast_dataset/",
    ):
        self.model_name = model_name
        self.config = config
        self.out_dataset_path = out_dataset_path
        self.dataloader = dataloader
        self.manual_seed = 999 # manualSeed = random.randint(1, 10000) # use if you want new results
        print("Random Seed: ", self.manual_seed)
        random.seed(self.manual_seed)
        torch.manual_seed(self.manual_seed)

        # Decide which device we want to run on
        self.device = torch.device("cuda" if (torch.cuda.is_available() and self.config.ngpu > 0) else "cpu")
        self._create_network()
        
    def _create_network(self):
        if self.model_name =="dcgan":
            if self.config.image_size == 64:
                if self.config.conditional:
                    from gan_compare.training.networks.dcgan.conditional.discriminator import Discriminator
                    from gan_compare.training.networks.dcgan.conditional.generator import Generator
                    assert self.config.nc == 2, "To use conditional input, change number of channels (nc) to 2."
                else:
                    from gan_compare.training.networks.dcgan.res64.discriminator import Discriminator
                    from gan_compare.training.networks.dcgan.res64.generator import Generator

                self.netG = Generator(
                    nz=self.config.nz, 
                    ngf=self.config.ngf, 
                    nc=self.config.nc, 
                    ngpu=self.config.ngpu,
                ).to(self.device)
                
                self.netD = Discriminator(
                    ndf=self.config.ndf, 
                    nc=self.config.nc, 
                    ngpu=self.config.ngpu,
                ).to(self.device)
                
            elif self.config.image_size == 128:
                from gan_compare.training.networks.dcgan.res128.discriminator import Discriminator
                from gan_compare.training.networks.dcgan.res128.generator import Generator

                self.netD = Discriminator(
                    ndf=self.config.ndf, 
                    nc=self.config.nc, 
                    ngpu=self.config.ngpu, 
                    leakiness=self.config.leakiness, 
                    bias=False,
                ).to(self.device)
                
                self.netG = Generator(                    
                    nz=self.config.nz, 
                    ngf=self.config.ngf, 
                    nc=self.config.nc, 
                    ngpu=self.config.ngpu,
                ).to(self.device)
            else:
                raise ValueError("Unsupported image size. Supported sizes are 128 and 64.")
        elif self.model_name == "lsgan":
            # only 64x64 image resolution will be supported
            assert self.config.image_size == 64, "Wrong image size for LSGAN, change it to 64x64 before proceeding."
            
            from gan_compare.training.networks.lsgan.discriminator import Discriminator
            from gan_compare.training.networks.lsgan.generator import Generator

            self.netG = Generator(                    
                    nz=self.config.nz, 
                    ngf=self.config.ngf, 
                    nc=self.config.nc, 
                    ngpu=self.config.ngpu,
            ).to(self.device)
            self.netD = Discriminator(
                    ndf=self.config.ndf, 
                    nc=self.config.nc, 
                    ngpu=self.config.ngpu, 
                    leakiness=self.config.leakiness, 
                    bias=False,
                ).to(self.device)
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")     
                # Handle multi-gpu if desired
        if (self.device.type == 'cuda') and (self.config.ngpu > 1):
           self.netG = nn.DataParallel(self.netG, list(range(self.config.ngpu)))  
        # Apply the weights_init function to randomly initialize all weights
        #  to mean=0, stdev=0.2.
        self.netG.apply(weights_init)

        # Print the model
        print(self.netG)
        
        # Handle multi-gpu if desired
        if (self.device.type == 'cuda') and (self.config.ngpu > 1):
            self.netD = nn.DataParallel(netD, list(range(self.config.ngpu)))
            
        # Apply the weights_init function to randomly initialize all weights
        #  to mean=0, stdev=0.2.
        self.netD.apply(weights_init)

        # Print the model
        print(self.netD) 

    def train(self):
        # Initialize BCELoss function
        self.criterion = nn.BCELoss()

        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        fixed_noise = torch.randn(12, self.config.nz, 1, 1, device=self.device)
        if self.config.conditional:
            fixed_conditon = torch.randint(self.config.birads_min, self.config.birads_max, (12,), device=self.device)

        # Establish convention for real and fake labels during training
        real_label = 1.
        fake_label = 0.

        # Setup Adam optimizers for both G and D
        optimizerD = optim.Adam(self.netD.parameters(), lr=self.config.lr, betas=(self.config.beta1, 0.999), weight_decay=self.config.weight_decay)
        optimizerG = optim.Adam(self.netG.parameters(), lr=self.config.lr, betas=(self.config.beta1, 0.999))
        
        # Training Loop

        # Lists to keep track of progress
        img_list = []
        G_losses = []
        D_losses = []
        iters = 0

        print("Starting Training Loop...")
        if self.config.conditional:
            print("Training conditioned on BiRADS")
        # For each epoch
        for epoch in range(self.config.num_epochs):
            # For each batch in the dataloader
            for i, data in enumerate(self.dataloader, 0):
                
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                self.netD.zero_grad()
                
                if self.config.conditional:
                    data, condition = data
                    # print(condition)
                
                # Format batch
                real_cpu = data[0].to(self.device)
                if self.config.conditional:
                    real_condition = condition.to(self.device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), real_label, dtype=torch.float, device=self.device)
                # Forward pass real batch through D
                if self.config.conditional:
                    output = self.netD(real_cpu, real_condition).view(-1)
                else:
                    output = self.netD(real_cpu).view(-1)
                # Calculate loss on all-real batch
                if self.model_name != "lsgan" and not self.config.use_lsgan_loss:
                    errD_real = self.criterion(output, label)
                else:
                    errD_real = 0.5 * torch.mean((output-label)**2)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, self.config.nz, 1, 1, device=self.device)
                
                # Generate fake image batch with G
                if self.config.conditional:
                    # generate fake conditions
                    fake_cond = torch.randint(self.config.birads_min, self.config.birads_max, (b_size,), device=self.device)
                    fake = self.netG(noise, fake_cond)
                else:
                    fake = self.netG(noise)
                label.fill_(fake_label)
                
                # Classify all fake batch with D
                if self.config.conditional:
                    output = self.netD(fake.detach(), fake_cond.detach()).view(-1)
                else:
                    output = self.netD(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                if self.model_name != "lsgan" and not self.config.use_lsgan_loss:
                    errD_fake = self.criterion(output, label)
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
                self.netG.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                if self.config.conditional:
                    output = self.netD(fake, fake_cond).view(-1)
                else:
                    output = self.netD(fake).view(-1)
                # Calculate G's loss based on this output
                if self.model_name != "lsgan" and not self.config.use_lsgan_loss:
                    errG = self.criterion(output, label)
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
                        % (epoch, self.config.num_epochs, i, len(self.dataloader),
                            errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                
                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())
                
                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 500 == 0) or ((epoch == self.config.num_epochs-1) and (i == len(self.dataloader)-1)):
                    with torch.no_grad():
                        if self.config.conditional:
                            fake = self.netG(fixed_noise, fixed_conditon).detach().cpu()
                        else:
                            fake = self.netG(fixed_noise).detach().cpu()
                    img_list.append(fake.numpy())
                    # img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                iters += 1
        output_model_dir = Path(self.config.output_model_dir)
        if self.config.conditional:
            out_path = output_model_dir / f"cond_{self.model_name}.pt"
        else:
            out_path = output_model_dir / f"{self.model_name}.pt"
        if not output_model_dir.exists():
            os.makedirs(output_model_dir.resolve())
        torch.save({
            "discriminator": self.netD.state_dict(), 
            "generator": self.netG.state_dict(), 
            "optim_discriminator": optimizerD.state_dict(), 
            "optim_generator": optimizerG.state_dict(),
        }, out_path)
        print(f"Saved model to {out_path.resolve()}")
        out_config_path = output_model_dir / f"config.yaml"
        save_yaml(path=out_config_path, data=self.config)
        print(f"Saved model config to {out_config_path.resolve()}")
        
        fig = plt.figure(figsize=(10,5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_losses,label="G")
        plt.plot(D_losses,label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        fig.savefig(str((output_model_dir / "training_progress.png").resolve()), dpi=fig.dpi)
        
        for i, image_batch in enumerate(img_list):
            for j, img_ in enumerate(image_batch):
                img_path = output_model_dir / f"{self.model_name}_{i}_{j}.png"
                img_ = interval_mapping(img_.transpose(1, 2, 0), -1., 1., 0, 255)
                img_ = img_.astype('uint8')
                cv2.imwrite(str(img_path.resolve()), img_)
            
    def generate(self, model_checkpoint_path: Path, num_samples: int = 64) -> list:
        optimizerD = optim.Adam(self.netD.parameters(), lr=self.config.lr, betas=(self.config.beta1, 0.999), weight_decay=self.config.weight_decay)
        optimizerG = optim.Adam(self.netG.parameters(), lr=self.config.lr, betas=(self.config.beta1, 0.999))

        checkpoint = torch.load(model_checkpoint_path)
        self.netD.load_state_dict(checkpoint['discriminator'])
        self.netG.load_state_dict(checkpoint['generator'])
        optimizerD.load_state_dict(checkpoint['optim_discriminator'])
        optimizerG.load_state_dict(checkpoint['optim_generator'])

        self.netG.eval()
        self.netD.eval()
        
        img_list = []
        for ind in range(num_samples):
            fixed_noise = torch.randn(num_samples, self.config.nz, 1, 1)
            if self.config.conditional:
                fixed_conditon = torch.randint(self.config.birads_min, self.config.birads_max, (num_samples,))
            if self.config.conditional:
                fake = self.netG(fixed_noise, fixed_conditon).detach().cpu().numpy()
            else:
                fake = self.netG(fixed_noise).detach().cpu().numpy()
            for j, img_ in enumerate(fake):
                img_list.extend(fake)
        return img_list
        