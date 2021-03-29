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

try:
    import tkinter
except:
    # Need to use matplotlib without tkinter dependency
    # tkinter is n.a. in some python distributions
    import matplotlib

    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
from typing import Union
from gan_compare.training.visualization import VisualizationUtils
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
        self.manual_seed = (
            999  # manualSeed = random.randint(1, 10000) # use if you want new results
        )
        print("Random Seed: ", self.manual_seed)
        random.seed(self.manual_seed)
        torch.manual_seed(self.manual_seed)

        # Decide which device we want to run on
        self.device = torch.device(
            "cuda" if (torch.cuda.is_available() and self.config.ngpu > 0) else "cpu"
        )
        self._create_network()

    def _create_network(self):
        if self.model_name == "dcgan":
            if self.config.image_size == 64:
                if self.config.conditional:
                    from gan_compare.training.networks.dcgan.conditional.discriminator import (
                        Discriminator,
                    )
                    from gan_compare.training.networks.dcgan.conditional.generator import (
                        Generator,
                    )

                    assert (
                            self.config.nc == 2
                    ), "To use conditional input, change number of channels (nc) to 2."

                    self.netG = Generator(
                        nz=self.config.nz,
                        ngf=self.config.ngf,
                        nc=self.config.nc,
                        ngpu=self.config.ngpu,
                        n_cond=self.config.n_cond
                    ).to(self.device)

                    self.netD = Discriminator(
                        ndf=self.config.ndf,
                        nc=self.config.nc,
                        ngpu=self.config.ngpu,
                        n_cond=self.config.n_cond
                    ).to(self.device)

                else:
                    from gan_compare.training.networks.dcgan.res64.discriminator import (
                        Discriminator,
                    )
                    from gan_compare.training.networks.dcgan.res64.generator import (
                        Generator,
                    )

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
                from gan_compare.training.networks.dcgan.res128.discriminator import (
                    Discriminator,
                )
                from gan_compare.training.networks.dcgan.res128.generator import (
                    Generator,
                )

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
                raise ValueError(
                    "Unsupported image size. Supported sizes are 128 and 64."
                )
        elif self.model_name == "lsgan":
            # only 64x64 image resolution will be supported
            assert (
                    self.config.image_size == 64
            ), "Wrong image size for LSGAN, change it to 64x64 before proceeding."

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
        if (self.device.type == "cuda") and (self.config.ngpu > 1):
            self.netG = nn.DataParallel(self.netG, list(range(self.config.ngpu)))
        # Apply the weights_init function to randomly initialize all weights
        #  to mean=0, stdev=0.2.
        self.netG.apply(weights_init)

        # Print the model
        print(self.netG)

        # Handle multi-gpu if desired
        if (self.device.type == "cuda") and (self.config.ngpu > 1):
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
        # TODO let's make the fixed batch size of 12 a config param(?)
        fixed_noise = torch.randn(12, self.config.nz, 1, 1, device=self.device)
        fixed_condition = None
        if self.config.conditional:
            fixed_condition = torch.randint(
                self.config.birads_min,
                self.config.birads_max,
                (12,),
                device=self.device,
            )

        # Establish convention for real and fake labels during training
        real_label = 1.0
        fake_label = 0.0

        # Setup Adam optimizers for both G and D
        optimizerD = optim.Adam(
            self.netD.parameters(),
            lr=self.config.lr,
            betas=(self.config.beta1, 0.999),
            weight_decay=self.config.weight_decay,
        )
        optimizerG = optim.Adam(
            self.netG.parameters(), lr=self.config.lr, betas=(self.config.beta1, 0.999)
        )
        # define the model storage directory
        output_model_dir = Path(self.config.output_model_dir)
        if not output_model_dir.exists():
            os.makedirs(output_model_dir.resolve())

        # visualize the model
        # initializing variables needed for visualization
        running_loss_of_generator = 0.
        running_loss_of_discriminator = 0.
        running_real_discriminator_accuracy = 0.
        running_fake_discriminator_accuracy = 0.
        # TODO move num_iterations_between_prints to config
        num_iterations_between_prints = 100
        # TODO move discriminator_clf_threshold to config (this variable is used to calculate discriminator accuracy
        discriminator_clf_threshold = 0.5
        # visualize model in tensorboard and instantiate visualizationUtils class object
        visualization_utils = self.visualize(output_model_dir=output_model_dir, fixed_noise=fixed_noise,
                                             fixed_condition=fixed_condition,
                                             num_iterations_between_prints=num_iterations_between_prints)

        # Lists to keep track of progress
        img_list = []
        G_losses = []
        D_losses = []
        iters = 0

        # Training Loop
        print("Starting Training.. ")
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
                label = torch.full(
                    (b_size,), real_label, dtype=torch.float, device=self.device
                )
                # Forward pass real batch through D
                if self.config.conditional:
                    output = self.netD(real_cpu, real_condition).view(-1)
                else:
                    output = self.netD(real_cpu).view(-1)
                # Calculate loss on all-real batch
                if self.model_name != "lsgan" and not self.config.use_lsgan_loss:
                    errD_real = self.criterion(output, label)
                else:
                    errD_real = 0.5 * torch.mean((output - label) ** 2)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()
                output_real = output
                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, self.config.nz, 1, 1, device=self.device)

                # Generate fake image batch with G
                if self.config.conditional:
                    # generate fake conditions
                    fake_cond = torch.randint(
                        self.config.birads_min,
                        self.config.birads_max,
                        (b_size,),
                        device=self.device,
                    )
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
                    errD_fake = 0.5 * torch.mean((output - label) ** 2)
                # Calculate the gradients for this batch
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                output_fake1 = output
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
                    errG = 0.5 * torch.mean((output - label) ** 2)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                output_fake2 = output
                # Update G
                optimizerG.step()

                # Calculate D's accuracy on the real data with real_label being = 1
                current_real_acc = torch.sum(output_real > discriminator_clf_threshold).item() / \
                                   list(output_real.size())[0]
                running_real_discriminator_accuracy += current_real_acc
                # print("REAL found by D#: " + str(torch.sum(output_real > discriminator_clf_threshold).item()))
                # print("REAL accuracy %: " + str(torch.sum(output_real > discriminator_clf_threshold).item() / list(output_real.size())[0]))

                # Calculate D's accuracy on the fake data from G with fake_label being = 0
                current_fake_acc = torch.sum(output_fake1 < discriminator_clf_threshold).item() / \
                                   list(output_fake1.size())[0]
                running_fake_discriminator_accuracy += current_fake_acc
                # print("FAKE found by D #: " + str(torch.sum(output_fake1 < discriminator_clf_threshold).item()))
                # print("FAKE accuracy %: " + str(torch.sum(output_fake1 < discriminator_clf_threshold).item() / list(output_fake1.size())[0]))

                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())

                # Update the running loss which is used in visualization
                running_loss_of_generator += errG.item()
                running_loss_of_discriminator += errD.item()

                # Output training stats on each iteration length threshold
                if i % num_iterations_between_prints == 0:
                    print(
                        '[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f\tAcc(D(x)): %.4f\tAcc(D(G(z)): %.4f'
                        % (epoch, self.config.num_epochs, i, len(self.dataloader),
                           errD.item(), errG.item(), D_x, D_G_z1, D_G_z2, current_real_acc, current_fake_acc))

                    # Add loss scalars to tensorboard
                    visualization_utils.add_value_to_tensorboard_loss_diagram(epoch=epoch,
                                                                              iteration=i,
                                                                              running_loss_of_generator=running_loss_of_generator,
                                                                              running_loss_of_discriminator=running_loss_of_discriminator)
                    # Add accuracy scalars to tensorboard
                    visualization_utils.add_value_to_tensorboard_accuracy_diagram(epoch=epoch,
                                                                                  iteration=i,
                                                                                  running_real_discriminator_accuracy=running_real_discriminator_accuracy,
                                                                                  running_fake_discriminator_accuracy=running_fake_discriminator_accuracy)
                    # Reset the running losses and accuracies
                    running_loss_of_generator = 0.
                    running_loss_of_discriminator = 0.
                    running_real_discriminator_accuracy = 0.
                    running_fake_discriminator_accuracy = 0.

                # Visually check how the generator is doing by saving G's output on fixed_noise
                if (iters % num_iterations_between_prints * 10 == 0) or (
                        (epoch == self.config.num_epochs - 1) and (i == len(self.dataloader) - 1)):
                    img_name: str = 'generated fixed-noise images each <' \
                                    + str(num_iterations_between_prints * 10) \
                                    + '>th iteration. Epoch=' \
                                    + str(epoch) \
                                    + ', iteration=' \
                                    + str(i)
                    visualization_utils.add_generated_batch_to_tensorboard(neural_network=self.netG,
                                                                           network_input_1=fixed_noise,
                                                                           network_input_2=fixed_condition,
                                                                           img_name=img_name)
                iters += 1
            visualization_utils.plot_losses(D_losses=D_losses, G_losses=G_losses)


        if self.config.conditional:
            out_path = output_model_dir / f"cond_{self.model_name}.pt"
        else:
            out_path = output_model_dir / f"{self.model_name}.pt"
        torch.save(
            {
                "discriminator": self.netD.state_dict(),
                "generator": self.netG.state_dict(),
                "optim_discriminator": optimizerD.state_dict(),
                "optim_generator": optimizerG.state_dict(),
            },
            out_path,
        )
        print(f"Saved model to {out_path.resolve()}")
        out_config_path = output_model_dir / f"config.yaml"
        save_yaml(path=out_config_path, data=self.config)
        print(f"Saved model config to {out_config_path.resolve()}")


    def generate(self, model_checkpoint_path: Path, num_samples: int = 64) -> list:
        optimizerD = optim.Adam(
            self.netD.parameters(),
            lr=self.config.lr,
            betas=(self.config.beta1, 0.999),
            weight_decay=self.config.weight_decay,
        )
        optimizerG = optim.Adam(
            self.netG.parameters(), lr=self.config.lr, betas=(self.config.beta1, 0.999)
        )

        checkpoint = torch.load(model_checkpoint_path)
        self.netD.load_state_dict(checkpoint["discriminator"])
        self.netG.load_state_dict(checkpoint["generator"])
        optimizerD.load_state_dict(checkpoint["optim_discriminator"])
        optimizerG.load_state_dict(checkpoint["optim_generator"])

        self.netG.eval()
        self.netD.eval()

        img_list = []
        for ind in range(num_samples):
            fixed_noise = torch.randn(num_samples, self.config.nz, 1, 1, device=self.device)
            if self.config.conditional:
                fixed_condition = torch.randint(
                    self.config.birads_min, self.config.birads_max, (num_samples,), device=self.device
                )
                fake = self.netG(fixed_noise, fixed_condition).detach().cpu().numpy()
            else:
                fake = self.netG(fixed_noise).detach().cpu().numpy()
            for j, img_ in enumerate(fake):
                img_list.extend(fake)
        return img_list


    def visualize(self, output_model_dir, fixed_noise=None, fixed_condition=None, batch_size: int = 12,
                  num_iterations_between_prints: int = 100):
        with torch.no_grad():
            # we need the number of training iterations per epoch (depending on size of batch and training dataset)
            num_iterations_per_epoch = len(self.dataloader)

            # Setup visualizaion utilities, which includes tensorboard I/O functions
            visualization_utils = VisualizationUtils(num_iterations_per_epoch=num_iterations_per_epoch,
                                                     num_iterations_between_prints=num_iterations_between_prints,
                                                     output_model_dir=output_model_dir)
            if fixed_noise is None:
                fixed_noise = torch.randn(batch_size, self.config.nz, 1, 1, requires_grad=False, device=self.device)

            if self.config.conditional and fixed_condition is None:
                fixed_condition = torch.randint(
                    self.config.birads_min,
                    self.config.birads_max,
                    (batch_size,),
                    requires_grad=False,
                    device=self.device
                )

            visualization_utils.generate_tensorboard_network_graph(neural_network=self.netG,
                                                                   network_input_1=fixed_noise,
                                                                   network_input_2=fixed_condition)

            visualization_utils.generate_tensorboard_network_graph(neural_network=self.netD,
                                                                   network_input_1=self.netG(fixed_noise,
                                                                                             fixed_condition),
                                                                   network_input_2=fixed_condition)
            return visualization_utils
