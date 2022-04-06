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
from gan_compare.training.networks.generation.dcgan.discriminator import (
    Discriminator,
)
from gan_compare.training.networks.generation.dcgan.generator import (
    Generator,
)


class DCGANModel(BaseGANModel):
    def __init__(
            self,
            config: GANConfig,
            dataloader: DataLoader
    ):
        super(DCGANModel, self).__init__(config=config, dataloader=dataloader)
        self.config = config
        self.dataloader = dataloader
        self._create_network()
        self.netD, self.netG = self.network_setup(netD=self.netD, netG=self.netG)
        self.optimizerD, self.optimizerG = self.optimizer_setup(netD=self.netD, netG=self.netG)
        self.loss = self._compute_switching_loss if self.config.switch_loss_each_epoch else self._compute_loss

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

    def _compute_switching_loss(
            self,
            output,
            label,
            epoch: int,
            are_outputs_logits: bool = False,
    ):
        """ Computing and returning the loss which swiches each epoch between ls and bce loss."""

        assert not are_outputs_logits, "Logits as output of the discriminator are not allowed when using switching loss. Please revise."

        if self.criterion is None:
            self.criterion = nn.BCELoss()

        # Note this design decision: switch_loss_each_epoch:bool=True overwrites use_lsgan_loss:bool=False
        if epoch % 2 == 0:
            # if epoch is even, we use BCE loss, if it is uneven, we use least square loss.
            logging.debug(
                f"switch_loss={self.config.switch_loss_each_epoch}, epoch={epoch}, epoch%2 = {epoch % 2} == 0 -> BCE loss."
            )
            return self.criterion(output, label)
        else:
            logging.debug(
                f"switch_loss={self.config.switch_loss_each_epoch}, epoch={epoch}, epoch%2 = {epoch % 2} != 0 -> LS loss."
            )
            return 0.5 * torch.mean((output - label) ** 2)

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
        labels_real = self._get_labels(b_size=output_real.size(0), label="real",
                                       smoothing=self.config.use_one_sided_label_smoothing, )
        labels_fake = self._get_labels(b_size=output_fake.size(0), label="fake", )

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
        return (
            output_real,
            errD_real,
            D_x,
            output_fake,
            errD_fake,
            D_G_z1,
            errD,
        )

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
        output, D_G_z2 = self._netD_forward_pass(netD=netD, images=fake_images, conditions=fake_conditions, )

        # Generate label is repeated each time due to varying b_size i.e. last batch of epoch has less images
        # Here, the "real" label is needed, as the fake labels are "real" for generator cost.
        # label smoothing is False, as this option would decrease the loss of the generator.
        labels = self._get_labels(b_size=output.size(0), label="real", smoothing=False, )

        # Calculate G's loss based on D's output
        errG = self.loss(
            output=output,
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

        return output, D_G_z2, errG

    def compute_discriminator_accuracy(
            self,
            output_real_D,
            running_real_D_acc,
            output_fake_D,
            running_fake_D_acc,
            output_real_D2=None,
            running_real_D2_acc=None,
            output_fake_D2=None,
            running_fake_D2_acc=None,
    ):
        """ compute the current training accuracy metric """

        # Calculate D's accuracy on the real data with real_label being = 1.
        # D1
        current_real_D_acc = (
                torch.sum(
                    output_real_D > self.config.discriminator_clf_threshold
                ).item()
                / list(output_real_D.size())[0]
        )

        # Calculate D's accuracy on the fake data from G with fake_label being = 0.
        # Note that we use the output_fake_D and not output_fake_2_D1, as 2 would be unfair,
        # as the discriminator has already received a weight update for the training batch
        current_fake_D_acc = (
                torch.sum(
                    output_fake_D < self.config.discriminator_clf_threshold
                ).item()
                / list(output_fake_D.size())[0]
        )

        running_real_D_acc += current_real_D_acc
        running_fake_D_acc += current_fake_D_acc

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

        return output_real_D, running_real_D_acc, output_fake_D, running_fake_D_acc, current_real_D_acc, current_fake_D_acc, output_real_D2, running_real_D2_acc, output_fake_D2, running_fake_D2_acc, current_real_D2_acc, current_fake_D2_acc

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
    ):

        """ logging the training progress and current metrics to console """

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

    def train(self):
        """ Training the GAN network iterating over the dataloader """

        # initializing variables needed for visualization and # lists to keep track of progress
        running_loss_of_generator, running_loss_of_discriminator, running_real_D_acc, running_fake_D_acc, G_losses, D_losses = self.init_running_losses()

        iters = 0

        # set to None for function calls later below
        running_loss_of_discriminator2, running_loss_of_generator_D2, running_real_D2_acc, running_fake_D2_acc, D2_losses, G2_losses = self.init_running_losses(
            init_value=0.0 if self.config.pretrain_classifier else None)

        # Training Loop
        logging.info(
            f"Starting Training on {self.device}. Image size: {self.config.image_size}"
        ) if not self.config.conditional else logging.info(
            f"Starting Training on {self.device}. Image size: {self.config.image_size}. Training conditioned on: {self.config.conditioned_on}"
            f"{' as a continuous (with random noise: ' + f'{self.config.added_noise_term}' + ') and not' * (1 - self.config.is_condition_categorical)} as a categorical variable"
        )

        # For each epoch
        for epoch in range(self.config.num_epochs):

            # We check if we should include D2 into training in the current epoch by resetting config.pretrain_classifier
            self.config.pretrain_classifier = self.is_D2_pretrained(epoch=epoch)

            # d_iteration: For each n discriminator ("critic") updates we do 1 generator update.
            d_iteration: int = (
                self.config.d_iters_per_g_update
            )  # In the very first iteration 0, G is trained.

            # For each batch in the dataloader
            for i, data in enumerate(self.dataloader, 0):

                # Unpack data (=image batch) alongside condition (e.g. birads number). Conditions are all -1 if unconditioned.
                try:
                    (
                        data,
                        conditions,
                        _,
                    ) = data
                except:
                    data, conditions, _, _ = data

                # Compute the actual batch size (not from config!) as last batch might be smaller than.
                b_size = data.size(0)

                # Format batch (fake and real), get images and, optionally, corresponding conditional GAN inputs
                real_images = data.to(self.device)

                real_conditions = None
                fake_conditions = None
                if self.config.conditional:
                    real_conditions = conditions.to(self.device)
                    # generate fake conditions
                    fake_images, fake_conditions = self._netG_forward_pass(
                        b_size=b_size
                    )
                else:
                    # Generate fake image batch with G (without condition)
                    fake_images, _ = self._netG_forward_pass(b_size=b_size)

                # We start by updating the discriminator
                # Reset the gradient of the discriminator of previous training iterations
                self.netD.zero_grad()

                # Perform a forward backward training step for D with optimizer weight update for real and fake data
                (
                    output_real_D,
                    errD_real,
                    D_x,
                    output_fake_D,
                    errD_fake,
                    D_G_z1,
                    errD,
                ) = self._netD_update(
                    netD=self.netD,
                    optimizerD=self.optimizerD,
                    real_images=real_images,
                    fake_images=fake_images.detach(),
                    epoch=epoch,
                    are_outputs_logits=False,
                    real_conditions=real_conditions,
                    fake_conditions=fake_conditions,
                )

                if self.config.pretrain_classifier:
                    self.netD2.zero_grad()
                    are_outputs_logits = (
                        True if self.config.model_name == "swin_transformer" else False
                    )
                    (
                        output_real_D2,
                        errD2_real,
                        D2_x,
                        output_fake_D2,
                        errD2_fake,
                        D2_G_z1,
                        errD2,
                    ) = self._netD_update(
                        netD=self.netD2,
                        optimizerD=self.optimizerD2,
                        real_images=real_images,
                        fake_images=fake_images.detach(),
                        epoch=epoch,
                        are_outputs_logits=are_outputs_logits,
                        real_conditions=real_conditions,
                        fake_conditions=fake_conditions,
                    )
                errG = None
                errG_D2 = None
                # After updating the discriminator, we now update the generator
                if not d_iteration == self.config.d_iters_per_g_update:
                    # We update D n times per G update. n = self.config.d_iters_per_g_update
                    d_iteration = d_iteration + 1

                else:

                    # Reset d_iteration to zero.
                    d_iteration = 0

                    # Reset to zero as previous gradient should have already been used to update the generator network
                    self.netG.zero_grad()

                    # (optional) Incomment if you want to generate new fake images as previous ones are already incorporated in D's gradient update
                    # fake_images, fake_conditions = self.generate_during_training(b_size=b_size)

                    # Perform a forward backward training step for G with optimizer weight update including a second
                    # output prediction by D to get bigger gradients as D has been already updated on this fake image batch.
                    (
                        output_fake_2_D1,
                        D_G_z2,
                        errG,
                        output_fake_2_D2,
                        D2_G_z2,
                        errG_D2,
                    ) = self.handle_G_updates(
                        iteration=i,
                        fake_images=fake_images,
                        fake_conditions=fake_conditions,
                        epoch=epoch,
                        b_size=b_size,
                        is_D2_using_new_fakes=False,  # TODO: Try is_D2_using_new_fakes out and see if it works better
                    )

                    # Save G Losses for plotting later
                    G_losses.append(errG.item())
                    # Update the running loss of G which is used in visualization
                    running_loss_of_generator += errG.item()

                    if self.config.pretrain_classifier:
                        G2_losses.append(errG_D2.item())
                        running_loss_of_generator_D2 += errG_D2.item()

                # Save D Losses for plotting later
                D_losses.append(errD.item())
                # Update the running loss which is used in visualization
                running_loss_of_discriminator += errD.item()

                if self.config.pretrain_classifier:
                    D2_losses.append(errD2.item())
                    running_loss_of_discriminator2 += errD2.item()

                # Accuracy of D1 (and D2) if they predict real/fake of synthetic images from G
                output_real_D, running_real_D_acc, output_fake_D, running_fake_D_acc, current_real_D_acc, current_fake_D_acc, output_real_D2, running_real_D2_acc, output_fake_D2, running_fake_D2_acc, current_real_acc_2, current_fake_acc_2 = self.compute_discriminator_accuracy(
                    output_real_D=output_real_D,
                    running_real_D_acc=running_real_D_acc,
                    output_fake_D=output_fake_D,
                    running_fake_D_acc=running_fake_D_acc,
                    output_real_D2= None if not self.config.pretrain_classifier else output_real_D2,
                    running_real_D2_acc= None if not self.config.pretrain_classifier else running_real_D2_acc,
                    output_fake_D2= None if not self.config.pretrain_classifier else output_fake_D2,
                    running_fake_D2_acc= None if not self.config.pretrain_classifier else running_fake_D2_acc,
                )

                # Output training stats on each iteration length threshold
                if i % self.config.num_iterations_between_prints == 0:
                    self.periodic_training_console_log(
                        epoch=epoch,
                        iteration=i,
                        errD=errD,
                        errG=errG,
                        D_x=D_x,
                        D_G_z1=D_G_z1,
                        D_G_z2=D_G_z2,
                        current_real_D_acc=current_real_D_acc,
                        current_fake_D_acc=current_fake_D_acc,
                        errD2= None if not self.config.pretrain_classifier else errD2,
                        errG_D2= None if not self.config.pretrain_classifier else errG_D2,
                        D2_x= None if not self.config.pretrain_classifier else D2_x,
                        D2_G_z1=None if not self.config.pretrain_classifier else D2_G_z1,
                        D2_G_z2=None if not self.config.pretrain_classifier else D2_G_z2,
                        current_real_acc_2= None if not self.config.pretrain_classifier else current_real_acc_2,
                        current_fake_acc_2= None if not self.config.pretrain_classifier else current_fake_acc_2,
                    )
                    self.periodic_visualization_log(
                        epoch=epoch,
                        iteration=i,
                        running_loss_of_generator=running_loss_of_generator,
                        running_loss_of_discriminator=running_loss_of_discriminator,
                        running_real_discriminator_accuracy=running_real_D_acc,
                        running_fake_discriminator_accuracy=running_fake_D_acc,
                        running_loss_of_generator_D2= None if not self.config.pretrain_classifier else running_loss_of_generator_D2,
                        running_loss_of_discriminator2= None if not self.config.pretrain_classifier else running_loss_of_discriminator2,
                        running_real_discriminator2_accuracy= None if not self.config.pretrain_classifier else running_real_D2_acc,
                        running_fake_discriminator2_accuracy= None if not self.config.pretrain_classifier else running_fake_D2_acc,
                    )

                    # Reset the running losses and accuracies
                    running_loss_of_generator, running_loss_of_discriminator, running_real_D_acc, running_fake_D_acc, _, _ = self.init_running_losses(
                        init_value=0.0)

                    if self.config.pretrain_classifier:
                        running_loss_of_discriminator2, running_loss_of_generator_D2, running_real_D2_acc, running_fake_D2_acc, _, _ = self.init_running_losses(
                            init_value=0.0)

                iters += 1
            self.visualization_utils.plot_losses(
                D_losses=D_losses,
                D2_losses=D2_losses,
                G_losses=G_losses,
                G2_losses=G2_losses,
            )
            if (
                    epoch % self.config.num_epochs_between_gan_storage == 0
                    and epoch >= self.config.num_epochs_before_gan_storage
            ):
                # Save on each {self.config.num_epochs_between_gan_storage}'th epoch starting at epoch {self.config.num_epochs_before_gan_storage}.
                self._save_model(epoch)
        self._save_model()
