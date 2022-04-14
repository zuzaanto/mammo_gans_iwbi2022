from __future__ import print_function

import logging

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.utils.data import DataLoader

try:
    import tkinter
except:
    # Need to use matplotlib without tkinter dependency
    # tkinter is n.a. in some python distributions
    import matplotlib

    matplotlib.use("Agg")

from gan_compare.training.gan_config import GANConfig
from gan_compare.training.networks.generation.base_gan_model import BaseGANModel
from gan_compare.training.networks.generation.dcgan.discriminator import Discriminator
from gan_compare.training.networks.generation.dcgan.generator import Generator


class DCGANModel(BaseGANModel):
    def __init__(self, config: GANConfig, dataloader: DataLoader):
        super(DCGANModel, self).__init__(config=config, dataloader=dataloader)
        self.config = config
        self.dataloader = dataloader
        self._create_network()
        self.netD, self.netG = self.network_setup(netD=self.netD, netG=self.netG)
        self.optimizerD, self.optimizerG = self.optimizer_setup(
            netD=self.netD, netG=self.netG
        )
        self.loss = (
            self._compute_switching_loss
            if self.config.switch_loss_each_epoch
            else self._compute_loss
        )

    def _create_network(self):
        """Importing and initializing the desired GAN architecture, weights and configuration."""

        self.netD = Discriminator(
            ndf=self.config.ndf,
            nc=self.config.nc,
            ngpu=self.config.ngpu,
            image_size=self.config.image_size,
            conditional=self.config.conditional,
            n_cond=self.config.n_cond,
            is_condition_categorical=self.config.is_condition_categorical,
            num_embedding_dimensions=self.config.num_embedding_dimensions,
            kernel_size=self.config.kernel_size,
            leakiness=self.config.leakiness,
        ).to(self.device)

        self.netG = Generator(
            nz=self.config.nz,
            ngf=self.config.ngf,
            nc=self.config.nc,
            ngpu=self.config.ngpu,
            image_size=self.config.image_size,
            conditional=self.config.conditional,
            n_cond=self.config.n_cond,
            is_condition_categorical=self.config.is_condition_categorical,
            num_embedding_dimensions=self.config.num_embedding_dimensions,
            leakiness=self.config.leakiness,
        ).to(self.device)

    def _compute_loss(
        self,
        output,
        label,
        epoch: int,
        are_outputs_logits: bool = False,
    ):
        """Setting the BCE loss function. Computing and returning the loss."""

        # Avoiding to reset self.criterion on each loss calculation
        is_output_type_changed = self.are_outputs_logits != are_outputs_logits
        self.are_outputs_logits = are_outputs_logits
        if (
            not hasattr(self, "criterion")
            or self.criterion is None
            or is_output_type_changed
        ):
            # Initialize standard criterion. Note: Could be moved to config.
            if are_outputs_logits:
                logging.debug(f"epoch {epoch}: Now using BCEWithLogitsLoss")
                self.criterion = nn.BCEWithLogitsLoss()
            else:
                logging.debug(f"epoch {epoch}: Now using BCELoss")
                self.criterion = nn.BCELoss()
        # Standard criterion defined above - e.g. Vanilla GAN's binary cross entropy (BCE) loss
        logging.debug(f"output: {output} \n label: {label}")
        return self.criterion(output, label)

    def _netD_update(
        self,
        netD,
        optimizerD,
        real_images,
        fake_images,
        epoch: int,
        real_conditions=None,
        fake_conditions=None,
        are_outputs_logits=False,
    ):
        """Update Discriminator network on real AND fake data."""

        # Forward pass real batch through D
        output_real, D_x = self._netD_forward_pass(
            netD,
            real_images,
            real_conditions,
        )

        # Forward pass fake batch through D
        output_fake, D_G_z1 = self._netD_forward_pass(
            netD,
            fake_images.detach(),
            fake_conditions,
        )
        # Create labels fo real and fake batches
        labels_real = self._get_labels(
            b_size=output_real.size(0),
            label="real",
            smoothing=self.config.use_one_sided_label_smoothing,
        )
        labels_fake = self._get_labels(
            b_size=output_fake.size(0),
            label="fake",
        )

        # Calculate D loss for fake batch
        errD_fake = self.loss(
            output=output_fake,
            label=labels_fake,
            epoch=epoch,
            are_outputs_logits=are_outputs_logits,
        )

        # Calculate D loss for real batch
        errD_real = self.loss(
            output=output_real,
            label=labels_real,
            epoch=epoch,
            are_outputs_logits=are_outputs_logits,
        )

        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake

        # Calculate gradients for D in backward pass
        errD.backward()

        # Update D
        optimizerD.step()
        return (output_real, D_x, output_fake, D_G_z1, errD, None)

    def _netG_update(
        self,
        netD,
        fake_images,
        fake_conditions,
        epoch: int,
        are_outputs_logits: bool = False,
        retain_graph: bool = False,
        is_G_updated: bool = True,
    ):
        """Update Generator network: e.g. in dcgan the goal is to maximize log(D(G(z)))"""

        # Since we just updated D, perform another forward pass of all-fake batch through the updated D.
        # The generator loss of the updated discriminator should be higher than the previous one.
        # D_G_z2 is the D output mean on the second generator input
        output_fake_2_D, D_G_z2 = self._netD_forward_pass(
            netD=netD,
            images=fake_images,
            conditions=fake_conditions,
        )

        # Generate label is repeated each time due to varying b_size i.e. last batch of epoch has less images
        # Here, the "real" label is needed, as the fake labels are "real" for generator cost.
        # label smoothing is False, as this option would decrease the loss of the generator.
        labels = self._get_labels(
            b_size=output_fake_2_D.size(0),
            label="real",
            smoothing=False,
        )

        # Calculate G's loss based on D's output
        errG = self.loss(
            output=output_fake_2_D,
            label=labels,
            epoch=epoch,
            are_outputs_logits=are_outputs_logits,
        )

        if is_G_updated:
            # Calculate gradients for G
            errG.backward(
                retain_graph=retain_graph
            )  # another call to this backward() will happen if we pretrain the classifier
            # Update G
            if not retain_graph:
                # if the graph is retained, there will be another generator update pass through the network.
                # We wait until the last one before running the optimizerG.step()
                self.optimizerG.step()

        return output_fake_2_D, D_G_z2, errG

    def periodic_training_console_log(
        self,
        epoch,
        iteration,
        errD,
        errG,
        D_x,
        D_G_z1,
        D_G_z2,
        current_real_D_acc,
        current_fake_D_acc,
        errD2=None,
        errG_D2=None,
        D2_x=None,
        D2_G_z1=None,
        D2_G_z2=None,
        current_real_acc_2=None,
        current_fake_acc_2=None,
        **kwargs,
    ):

        """logging the training progress and current metrics to console"""

        if self.config.pretrain_classifier:
            # While not necessarily backpropagating into G, both D1 and D2 are used and we have all possible numbers available.
            logging.info(
                "[%d/%d][%d/%d]\tLoss_D1: %.4f\tLoss_D2: %.4f\tLoss_G_D1: %.4f\tLoss_G_D2: %.4f\tD(x): %.4f\tD(G(z1)): %.4f\tD(G(z2)): %.4f \tAcc(D(x)): %.4f\tAcc(D(G(z)): %.4f\tD2(x): %.4f\tD2(G(z1)): %.4f\tD2(G(z2)): %.4f \tAcc(D2(x)): %.4f\tAcc(D2(G(z)): %.4f"
                % (
                    epoch,
                    self.config.num_epochs - 1,
                    iteration,
                    len(self.dataloader),
                    errD.item(),
                    errD2.item(),
                    errG.item(),
                    errG_D2.item(),
                    D_x,
                    D_G_z1,
                    D_G_z2,
                    current_real_D_acc,
                    current_fake_D_acc,
                    D2_x,
                    D2_G_z1,
                    D2_G_z2,
                    current_real_acc_2,
                    current_fake_acc_2,
                )
            )
        else:
            # We only log D1 and G statistics, as D2 was not used in GAN training.
            logging.info(
                "[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z1)): %.4f\tD(G(z2)): %.4f\tAcc(D(x)): %.4f\tAcc(D(G(z)): %.4f"
                % (
                    epoch,
                    self.config.num_epochs - 1,
                    iteration,
                    len(self.dataloader),
                    errD.item(),
                    errG.item(),
                    D_x,
                    D_G_z1,
                    D_G_z2,
                    current_real_D_acc,
                    current_fake_D_acc,
                )
            )
