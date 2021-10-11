from __future__ import print_function

import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader

try:
    import tkinter
except:
    # Need to use matplotlib without tkinter dependency
    # tkinter is n.a. in some python distributions
    import matplotlib

    matplotlib.use("Agg")
from typing import Union, Optional
from gan_compare.training.visualization import VisualizationUtils
from gan_compare.training.gan_config import GANConfig
from gan_compare.training.networks.dcgan.utils import weights_init
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
        self.output_model_dir = Path(self.config.output_model_dir)

    def _mkdir_model_dir(self):
        if not self.output_model_dir.exists():
            os.makedirs(self.output_model_dir.resolve())

    def _save_config(self, config_file_name: str = f"config.yaml"):
        self._mkdir_model_dir()  # validation to make sure model dir exists
        out_config_path = self.output_model_dir / config_file_name
        save_yaml(path=out_config_path, data=self.config)
        print(f"Saved model config to {out_config_path.resolve()}")

    def _save_model(self, epoch_number: Optional[int] = None):
        self._mkdir_model_dir()  # validation to make sure model dir exists
        if epoch_number is None:
            out_path = self.output_model_dir / "model.pt"
        else:
            out_path = self.output_model_dir / f"{epoch_number}.pt"
        torch.save(
            {
                "discriminator": self.netD.state_dict(),
                "generator": self.netG.state_dict(),
                "optim_discriminator": self.optimizerD.state_dict(),
                "optim_generator": self.optimizerG.state_dict(),
            },
            out_path,
        )
        print(f"Saved model (on epoch(?): {epoch_number}) to {out_path.resolve()}")

    def _create_network(self):
        if self.model_name == "dcgan":
            print(f"self.config.kernel_size: {self.config.kernel_size}")
            print(f"self.config.image_size: {self.config.image_size}")

            from gan_compare.training.networks.dcgan.discriminator import (
                Discriminator
            )
            self.netD = Discriminator(
                ndf=self.config.ndf,
                nc=self.config.nc,
                ngpu=self.config.ngpu,
                image_size=self.config.image_size,
                is_conditional=self.config.conditional,
                n_cond=self.config.n_cond,
                is_condition_categorical=self.config.is_condition_categorical,
                num_embedding_dimensions=self.config.num_embedding_dimensions,
                kernel_size=self.config.kernel_size,
            ).to(self.device)

            from gan_compare.training.networks.dcgan.generator import (
                Generator
            )
            self.netG = Generator(
                nz=self.config.nz,
                ngf=self.config.ngf,
                nc=self.config.nc,
                ngpu=self.config.ngpu,
                image_size=self.config.image_size,
                is_conditional=self.config.conditional,
                n_cond=self.config.n_cond,
                is_condition_categorical=self.config.is_condition_categorical,
                num_embedding_dimensions=self.config.num_embedding_dimensions,
            ).to(self.device)

        elif self.model_name == "lsgan":
            # only 64x64 image resolution will be supported
            assert (
                self.config.image_size == 64
            ), "Wrong image size for LSGAN, change it to 64x64 before proceeding."
            assert (
                self.config.conditional
            ), "LSGAN does not support conditional inputs. Change conditional to False before proceeding."

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

        # Check if channel size is correct
        if self.config.conditional:
            assert (
                    self.config.nc == 2
            ), "To use conditional input, change number of channels (nc) to 2."
        else:
            assert (
                    self.config.nc == 1
            ), "Without conditional input into GAN, change number of channels (nc) to 1."

        # Handle multi-gpu if desired
        if (self.device.type == "cuda") and (self.config.ngpu > 1):
            self.netG = nn.DataParallel(self.netG, list(range(self.config.ngpu)))
        # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
        self.netG.apply(weights_init)

        # Print the generator model
        print(self.netG)

        # Handle multi-gpu if desired
        if (self.device.type == "cuda") and (self.config.ngpu > 1):
            self.netD = nn.DataParallel(self.netD, list(range(self.config.ngpu)))

        # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
        self.netD.apply(weights_init)

        # Print the discriminator model
        print(self.netD)


    def _netG_update(self, fake_images, fake_conditions, epoch: int):
        ''' Update Generator network: maximize log(D(G(z))) '''

        # Generate label is repeated each time due to varying b_size i.e. last batch of epoch has less images
        # Here, the "real" label is needed, as the fake labels are "real" for generator cost.
        # label smoothing is False, as this option would decrease the loss of the generator.
        labels = torch.full((fake_images.size(0),), self._get_labels(smoothing=False).get('real'), dtype=torch.float,
                            device=self.device)

        # Since we just updated D, perform another forward pass of all-fake batch through the updated D.
        # The generator loss of the updated discriminator should be higher than the previous one.
        if self.config.conditional:
            output = self.netD(fake_images, fake_conditions).view(-1)
        else:
            output = self.netD(fake_images).view(-1)

        # Calculate G's loss based on this output
        errG = self._compute_loss(output=output, label=labels, epoch=epoch)

        # Calculate gradients for G
        errG.backward()

        # D output mean on the second generator input
        D_G_z2 = output.mean().item()

        # Update G
        self.optimizerG.step()

        return output, D_G_z2, errG

    def _netD_update(self, real_images, fake_images, epoch: int, real_conditions=None, fake_conditions=None):
        ''' Update Discriminator network on real AND fake data. '''

        # Forward pass real batch through D
        output_real, errD_real, D_x = self._netD_forward_backward_pass(
            real_images,
            self._get_labels().get('real'),
            real_conditions,
            epoch=epoch,
        )
        # remove gradient only if fake_conditions is a torch tensor
        if fake_conditions is not None:
            fake_conditions = fake_conditions.detach()

        # Forward pass fake batch through D
        output_fake, errD_fake, D_G_z1 = self._netD_forward_backward_pass(
            fake_images.detach(),
            self._get_labels().get('fake'),
            fake_conditions,
            epoch=epoch,
        )

        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake

        # Update D
        self.optimizerD.step()

        return output_real, errD_real, D_x, output_fake, errD_fake, D_G_z1, errD

    def _netD_forward_backward_pass(self, images, label_as_float, conditions, epoch: int):
        ''' Forward and backward pass through discriminator network '''
        # Forward pass batch through D
        output = None
        if self.config.conditional:
            output = self.netD(images, conditions).view(-1)
        else:
            output = self.netD(images).view(-1)

        # Generate label is repeated each time due to varying b_size i.e. last batch of epoch has less images
        labels = torch.full((images.size(0),), label_as_float, dtype=torch.float, device=self.device)

        # Calculate loss on all-real batch
        errD = self._compute_loss(output=output, label=labels, epoch=epoch)

        # Calculate gradients for D in backward pass of real data batch
        errD.backward()
        D_input = output.mean().item()

        return output, errD, D_input

    def _compute_loss(self, output, label, epoch: int):
        if not hasattr(self, 'criterion') or self.criterion is None:
            # Initialize standard criterion. Note: Could be moved to config.
            self.criterion = nn.BCELoss()
        if self.config.switch_loss_each_epoch:
            # Note this design decision: switch_loss_each_epoch:bool=True overwrites use_lsgan_loss:bool=False
            if epoch % 2 == 0:
                # if epoch is even, we use BCE loss, if it is uneven, we use least square loss.
                # print(f'switch_loss={self.config.switch_loss_each_epoch}, epoch={epoch}, epoch%2 = {epoch % 2} == 0 '
                #      f'-> BCE loss.')
                return self.criterion(output, label)
            else:
                # print(f'switch_loss={self.config.switch_loss_each_epoch}, epoch={epoch}, epoch%2 = {epoch % 2} != 0 '
                #      f'-> LS loss.')
                return 0.5 * torch.mean((output - label) ** 2)
        else:
            if self.model_name != "lsgan" and not self.config.use_lsgan_loss:
                # Standard criterion defined above - i.e. Vanilla GAN's binary cross entropy (BCE) loss
                return self.criterion(output, label)
            else:
                # Least Square Loss - https://arxiv.org/abs/1611.04076
                return 0.5 * torch.mean((output - label) ** 2)

    def _get_labels(self, smoothing: bool = True):
        # if enabled, let's smooth the labels for "real" (--> real !=1)
        if self.config.use_one_sided_label_smoothing and smoothing:
            smoothed_real_label: float = random.uniform(self.config.label_smoothing_start,
                                                        self.config.label_smoothing_end)
            # print(f"smoothed_real_label = {smoothed_real_label}")
            return {"real": smoothed_real_label, "fake": 0.0}
        return {"real": 1.0, "fake": 0.0}

    def train(self):

        # As we train a new model (atm no continued checkpoint training), we create new model dir and save config.
        self._save_config()

        # Create batch of latent vectors that we will use to visualize the progression of the generator
        # Fpr convenience, let's use the specified batch size = number of fixed noise random tensors
        fixed_noise = torch.randn(self.config.batch_size, self.config.nz, 1, 1, device=self.device)

        # Create batch of fixed conditions that we will use to visualize the progression of the generator
        fixed_condition = None
        if self.config.conditional:
            fixed_condition = torch.randint(
                self.config.birads_min,
                self.config.birads_max + 1,
                (self.config.batch_size,),
                device=self.device,
            )

        # Setup Adam optimizers for both G and D
        self.optimizerD = optim.Adam(
            self.netD.parameters(),
            lr=self.config.lr,
            betas=(self.config.beta1, 0.999),
            weight_decay=self.config.weight_decay,
        )
        self.optimizerG = optim.Adam(
            self.netG.parameters(), lr=self.config.lr, betas=(self.config.beta1, 0.999)
        )

        # initializing variables needed for visualization
        running_loss_of_generator = 0.
        running_loss_of_discriminator = 0.
        running_real_discriminator_accuracy = 0.
        running_fake_discriminator_accuracy = 0.

        # visualize model in tensorboard and instantiate visualizationUtils class object
        visualization_utils = self.visualize(fixed_noise=fixed_noise, fixed_condition=fixed_condition)

        # Lists to keep track of progress
        G_losses = []
        D_losses = []
        iters = 0

        # Training Loop
        print(f"Starting Training.. Image size: {self.config.image_size}")
        if self.config.conditional:
            print(
                f"Training conditioned on BiRADS"
                f"{' as a continuous and not' * (1 - self.config.is_condition_categorical)} as a categorical variable")
        # For each epoch
        for epoch in range(self.config.num_epochs):
            # For each batch in the dataloader
            for i, data in enumerate(self.dataloader, 0):
                # We start by updating the discriminator
                # Reset the gradient of the discriminator of previous training iterations
                self.netD.zero_grad()

                # If the GAN has a conditional input, get condition (i.e. birads number) alongside data (=image batch)
                if self.config.conditional:
                    data, condition = data

                # Format batch (fake and real), get images and, optionally, corresponding conditional GAN inputs
                real_images = data[0].to(self.device)

                # Compute the actual batch size (not from config!) for convenience
                b_size = real_images.size(0)

                # Generate batch of latent vectors as input into generator to generate fake images
                noise = torch.randn(b_size, self.config.nz, 1, 1, device=self.device)

                real_conditions = None
                fake_conditions = None
                if self.config.conditional:
                    real_conditions = condition.to(self.device)
                    # generate fake conditions
                    fake_conditions = torch.randint(
                        self.config.birads_min,
                        self.config.birads_max + 1,
                        (b_size,),
                        device=self.device,
                    )
                    # Generate fake image batch with G (conditional_res64)
                    fake_images = self.netG(noise, fake_conditions)
                else:
                    # Generate fake image batch with G (without condition)
                    fake_images = self.netG(noise)

                # Perform a forward backward training step for D with optimizer weight update for real and fake data
                output_real, errD_real, D_x, output_fake_1, errD_fake, D_G_z1, errD = self._netD_update(
                    real_images=real_images,
                    fake_images=fake_images,
                    epoch=epoch,
                    real_conditions=real_conditions,
                    fake_conditions=fake_conditions,
                )

                # After updating the discriminator, we now update the generator
                # Reset to zero as previous gradient should have already been used to update the generator network
                self.netG.zero_grad()

                # Perform a forward backward training step for G with optimizer weight update including a second
                # output prediction by D to get bigger gradients as D has been already updated on this fake image batch.
                output_fake_2, D_G_z2, errG = self._netG_update(
                    fake_images=fake_images,
                    fake_conditions=fake_conditions,
                    epoch=epoch,
                )

                # Calculate D's accuracy on the real data with real_label being = 1.
                current_real_acc = torch.sum(output_real > self.config.discriminator_clf_threshold).item() / \
                                   list(output_real.size())[0]
                running_real_discriminator_accuracy += current_real_acc

                # Calculate D's accuracy on the fake data from G with fake_label being = 0.
                # Note that we use the output_fake_1 and not output_fake_2, as 2 would be unfair,
                # as the discriminator has already received a weight update for the training batch
                current_fake_acc = torch.sum(output_fake_1 < self.config.discriminator_clf_threshold).item() / \
                                   list(output_fake_1.size())[0]
                running_fake_discriminator_accuracy += current_fake_acc

                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())

                # Update the running loss which is used in visualization
                running_loss_of_generator += errG.item()
                running_loss_of_discriminator += errD.item()

                # Output training stats on each iteration length threshold
                if i % self.config.num_iterations_between_prints == 0:
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
                if (iters % self.config.num_iterations_between_prints * 10 == 0) or (
                        (epoch == self.config.num_epochs - 1) and (i == len(self.dataloader) - 1)):
                    img_name: str = 'generated fixed-noise images each <' \
                                    + str(self.config.num_iterations_between_prints * 10) \
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
            if (epoch % 5 == 0 and epoch >= 50):
                # Save on each 5th epoch starting at epoch 50.
                self._save_model(epoch)
        self._save_model()

    def generate(self, model_checkpoint_path: Path, fixed_noise=None, fixed_condition=None,
                 num_samples: int = 64, birads: int = None) -> list:

        self.optimizerG = optim.Adam(
            self.netG.parameters(), lr=self.config.lr, betas=(self.config.beta1, 0.999)
        )
        checkpoint = torch.load(model_checkpoint_path)
        self.netG.load_state_dict(checkpoint["generator"])
        self.optimizerG.load_state_dict(checkpoint["optim_generator"])
        self.netG.eval()

        img_list = []
        for ind in range(num_samples):
            if fixed_noise is None:
                fixed_noise = torch.randn(num_samples, self.config.nz, 1, 1, device=self.device)
            if self.config.conditional:
                if fixed_condition is None:
                    fixed_condition = torch.randint(
                        self.config.birads_min, self.config.birads_max + 1, (num_samples,), device=self.device
                    )
                elif isinstance(fixed_condition, int):
                    fixed_condition = torch.randint(
                        fixed_condition, fixed_condition + 1, (num_samples,), device=self.device
                    )
                # print(f'Tensor with batch of BIRADS conditions = {fixed_condition}')
                fake = self.netG(fixed_noise, fixed_condition).detach().cpu().numpy()
            else:
                fake = self.netG(fixed_noise).detach().cpu().numpy()
            for j, img_ in enumerate(fake):
                img_list.extend(fake)
        return img_list

    def visualize(self, fixed_noise=None, fixed_condition=None):
        with torch.no_grad():
            # we need the number of training iterations per epoch (depending on size of batch and training dataset)
            num_iterations_per_epoch = len(self.dataloader)

            # Setup visualizaion utilities, which includes tensorboard I/O functions
            visualization_utils = VisualizationUtils(num_iterations_per_epoch=num_iterations_per_epoch,
                                                     num_iterations_between_prints=self.config.num_iterations_between_prints,
                                                     output_model_dir=self.output_model_dir)
            if fixed_noise is None:
                fixed_noise = torch.randn(self.config.batch_size, self.config.nz, 1, 1, requires_grad=False,
                                          device=self.device)

            if self.config.conditional and fixed_condition is None:
                fixed_condition = torch.randint(
                    self.config.birads_min,
                    self.config.birads_max + 1,
                    (self.config.batch_size,),
                    requires_grad=False,
                    device=self.device
                )

            # Visualize the model architecture of the generator
            visualization_utils.generate_tensorboard_network_graph(
                neural_network=self.netG,
                network_input_1=fixed_noise,
                network_input_2=fixed_condition)
            return visualization_utils
