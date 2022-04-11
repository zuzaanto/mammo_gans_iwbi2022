from __future__ import print_function

import logging

import torch
import torch.nn.parallel
from torch import autograd, nn
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
from gan_compare.training.networks.generation.wgangp.discriminator import (
    Discriminator,
)
from gan_compare.training.networks.generation.wgangp.generator import (
    Generator,
)


class WGANGPModel(BaseGANModel):
    def __init__(
            self,
            config: GANConfig,
            dataloader: DataLoader
    ):
        super(WGANGPModel, self).__init__(config=config, dataloader=dataloader)
        self.config = config
        self.dataloader = dataloader
        self._create_network()
        self.netD, self.netG = self.network_setup(netD=self.netD, netG=self.netG)
        self.optimizerD, self.optimizerG = self.optimizer_setup(netD=self.netD, netG=self.netG)

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

    def _compute_loss_G(
            self,
            output_fake_2_D: str,
            are_outputs_logits: bool = False,
    ):
        """ Computing and returning the  WGAN-GP Generator loss. """

        assert not are_outputs_logits, "wgangp-loss does not yet work with logits as input. Please extend before using this function."
        return -output_fake_2_D.mean()  # G loss is the output from D i.e. d_fake_images

    def _compute_loss_D(
            self,
            netD,
            real_images,
            fake_images,
            are_outputs_logits: bool = False,
    ):
        """ Computing and returning the  WGAN-GP Discriminator loss. """

        assert not are_outputs_logits, "wgangp-loss does not yet work with logits as input. Please extend before using this function."
        assert (
                netD is not None
                and real_images is not None
                and fake_images is not None
        ), f"Please make sure that none of the variables 'fake_images', 'real_images', and 'netD' is None. Currently they are real_images={real_images}, fake_images={fake_images}, netD={netD}."

        gradient_penalty = self.compute_gradient_penalty(
            netD=netD,
            real_images=real_images,
            fake_images=fake_images,
        )
        output_fake_D, output_fake_D_mean = self._netD_forward_pass(netD=netD, images=fake_images)
        output_real_D, output_real_D_mean = self._netD_forward_pass(netD=netD, images=real_images)
        errD = (
                output_fake_D.mean()
                - output_real_D.mean()
                + gradient_penalty
        )  # D loss

        return (
            errD,
            output_fake_D,
            output_real_D,
            output_fake_D_mean,
            output_real_D_mean,
            gradient_penalty,
        )


    def compute_gradient_penalty(
            self, netD, real_images, fake_images,
    ):
        """gradient penalty computation according to paper https://arxiv.org/pdf/1704.00028.pdf

        Adapted from https://github.com/caogang/wgan-gp/blob/ae47a185ed2e938c39cf3eb2f06b32dc1b6a2064/gan_cifar10.py#L74
        """

        # Determine image shape
        b_size = real_images.size()[0]

        # Define alpha, which is a random number a ∼ U[0, 1].
        alpha = torch.rand(b_size, 1, 1, 1)
        alpha = alpha.expand_as(real_images)
        alpha = alpha.to(self.device)

        # Compute the x_hat as a random spread between real and generated
        x_hat = alpha * real_images + ((1 - alpha) * fake_images)
        x_hat.to(self.device)
        x_hat = autograd.Variable(x_hat, requires_grad=True)

        # Pass x_hat as a random spread between real and generated
        disc_x_hat, _ = self._netD_forward_pass(netD=netD, images=x_hat)

        # Compute gradient for disc_x_hat
        gradients = autograd.grad(
            outputs=disc_x_hat,
            inputs=x_hat,
            grad_outputs=torch.ones(disc_x_hat.size()).to(self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)

        # Compute the final gradient penalty where _lambda is the gradient penalty coefficient
        gradient_penalty = self.config.wgangp_lambda * (
            ((gradients.norm(2, dim=1) - 1) ** 2).mean()  # .item()
        )
        return gradient_penalty

    def _netD_update(
            self,
            netD,
            optimizerD,
            real_images,
            fake_images,
            epoch: int = None,
            are_outputs_logits=False,
            **kwargs,
    ):
        """Update WGANGP Discriminator network."""

        errD, output_fake_D, output_real_D, D_x, D_G_z1, gradient_penalty = self._compute_loss_D(
            netD=netD,
            real_images=real_images,
            fake_images=fake_images,
            are_outputs_logits=are_outputs_logits,
        )
        errD.backward()

        # Update D
        optimizerD.step()

        return (
            output_real_D,
            D_x,
            output_fake_D,
            D_G_z1,
            errD,
            gradient_penalty,
        )


    def _netG_update(
            self,
            netD,
            fake_images,
            fake_conditions = None,
            are_outputs_logits: bool = False,
            retain_graph: bool = False,
            is_G_updated: bool = True,
            **kwargs
    ):
        """Update the generator network"""

        # Since we just updated D, perform another forward pass of all-fake batch through the updated D.
        # The generator loss of the updated discriminator should be higher than the previous one.
        # D_G_z2 is the D output mean on the second generator input
        output_fake_2_D, D_G_z2 = self._netD_forward_pass(netD=netD, images=fake_images, conditions=fake_conditions, )

        # Calculate G's loss based on D's output
        errG = self._compute_loss_G(
            output_fake_2_D=output_fake_2_D,
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
            gradient_penalty,
            errD2=None,
            errG_D2=None,
            D2_x=None,
            D2_G_z1=None,
            D2_G_z2=None,
            current_real_acc_2=None,
            current_fake_acc_2=None,
            **kwargs,
    ):

        """ logging the training progress and current metrics to console """
        if self.config.pretrain_classifier:
            # While not necessarily backpropagating into G, both D1 and D2 are used and we have all possible numbers available.
            logging.info(
                "[%d/%d][%d/%d] \t Loss_D: %.4f \t Loss_G: %.4f \t D(x): %.4f \t D(G(z1)): %.4f \t D(G(z2)) %.4f \t Gradient Penalty: %.4f \t D2(x): %.4f\tD2(G(z1)): %.4f\tD2(G(z2)): %.4f \tAcc(D2(x)): %.4f\tAcc(D2(G(z)): %.4f"
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
                    gradient_penalty,
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
                "[%d/%d][%d/%d] \t Loss_D: %.4f \t Loss_G: %.4f \t D(x): %.4f \t D(G(z1)): %.4f \t D(G(z2)) %.4f \t Gradient Penalty: %.4f"
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
                    gradient_penalty,
                )
            )

    def compute_discriminator_accuracy(
            self,
            output_real_D2=None,
            running_real_D2_acc=None,
            output_fake_D2=None,
            running_fake_D2_acc=None,
            **kwargs
    ):
        """ compute the current training accuracy metric

        As discriminator (D1) in wgangp is not a classifier, we only calculate the accuracy for the second discriminator (D2)
        """

        # D2
        current_real_D2_acc = None if output_real_D2 is None else (
                torch.sum(
                    output_real_D2 > self.config.discriminator_clf_threshold
                ).item()
                / list(output_real_D2.size())[0]
        )
        current_fake_D2_acc = None if output_fake_D2 is None else (
                torch.sum(
                    output_fake_D2 < self.config.discriminator_clf_threshold
                ).item()
                / list(output_fake_D2.size())[0]
        )
        running_real_D2_acc = running_real_D2_acc if current_real_D2_acc is None else running_real_D2_acc + current_real_D2_acc
        running_fake_D2_acc = running_fake_D2_acc if current_fake_D2_acc is None else running_fake_D2_acc + current_fake_D2_acc

        return None, None, None, None, running_real_D2_acc, running_fake_D2_acc, current_real_D2_acc, current_fake_D2_acc

