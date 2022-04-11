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
from gan_compare.training.networks.generation.dcgan.dcgan_model import DCGANModel
from gan_compare.training.networks.generation.lsgan.discriminator import Discriminator
from gan_compare.training.networks.generation.lsgan.generator import Generator


class LSGANModel(DCGANModel):
    def __init__(self, config: GANConfig, dataloader: DataLoader):
        super(LSGANModel, self).__init__(config=config, dataloader=dataloader)
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

        # only 64x64 image resolution will be supported
        assert (
            self.config.image_size == 64
        ), "Wrong image size for LSGAN, change it to 64x64 before proceeding."
        assert (
            self.config.conditional == False
        ), "LSGAN does not support conditional inputs. Change conditional to False before proceeding."

    def _compute_loss(
        self,
        output,
        label,
        epoch=None,
        are_outputs_logits: bool = False,
    ):
        """Setting the LS loss function. Computing and returning the loss."""
        # Least Square Loss - https://arxiv.org/abs/1611.04076
        logging.debug(f"output: {output} \n label: {label}")
        if not are_outputs_logits:
            return 0.5 * torch.mean((output - label) ** 2)
        else:
            raise Exception(
                f"epoch={epoch}: ls-loss does not yet work with logits as input. Please extend before using this function"
            )

    def periodic_training_console_log(
        self,
        epoch,
        iteration,
        errD,
        errG,
        D_x,
        D_G_z1,
        D_G_z2,
        errD2=None,
        errG_D2=None,
        D2_x=None,
        D2_G_z1=None,
        D2_G_z2=None,
        **kwargs,
    ):
        """logging the training progress and current metrics to console"""

        if self.config.pretrain_classifier:
            # While not necessarily backpropagating into G, both D1 and D2 are used and we have all possible numbers available.
            logging.info(
                "[%d/%d][%d/%d]\tLoss_D1: %.4f\tLoss_D2: %.4f\tLoss_G_D1: %.4f\tLoss_G_D2: %.4f\tD(x): %.4f\tD(G(z1)): %.4f\tD(G(z2)): %.4f\tD2(x): %.4f\tD2(G(z1)): %.4f\tD2(G(z2)): %.4f "
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
                    D2_x,
                    D2_G_z1,
                    D2_G_z2,
                )
            )
        else:
            # We only log D1 and G statistics, as D2 was not used in GAN training.
            logging.info(
                "[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z1)): %.4f\tD(G(z2)): %.4f"
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
                )
            )
