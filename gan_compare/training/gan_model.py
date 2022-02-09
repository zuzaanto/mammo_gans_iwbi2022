from __future__ import print_function

import logging
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
from gan_compare.dataset.constants import DENSITY_DICT
from gan_compare.constants import get_classifier


class GANModel:
    def __init__(
            self,
            model_name: str,
            # TODO: Change naming convention. This should be "gan_type". config.model_name already in use -> This could cause confusion.
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
        logging.debug("Random Seed: ", self.manual_seed)
        random.seed(self.manual_seed)
        torch.manual_seed(self.manual_seed)

        # Decide which device we want to run on
        self.device = torch.device(
            "cuda" if (torch.cuda.is_available() and self.config.ngpu > 0) else "cpu"
        )
        self._create_network()
        self.output_model_dir = Path(self.config.output_model_dir)
        self.are_outputs_logits = None  # Outputs of D can be probabilities or logits. We need to handle both cases.

    def _mkdir_model_dir(self):
        if not self.output_model_dir.exists():
            os.makedirs(self.output_model_dir.resolve())

    def _save_config(self, config_file_name: str = f"config.yaml"):
        self._mkdir_model_dir()  # validation to make sure model dir exists
        out_config_path = self.output_model_dir / config_file_name
        save_yaml(path=out_config_path, data=self.config)
        logging.info(f"Saved model config to {out_config_path.resolve()}")

    def _save_model(self, epoch_number: Optional[int] = None):
        self._mkdir_model_dir()  # validation to make sure model dir exists
        if epoch_number is None:
            out_path = self.output_model_dir / "model.pt"
        else:
            out_path = self.output_model_dir / f"{epoch_number}.pt"

        d = {
            "discriminator": self.netD.state_dict(),
            "generator": self.netG.state_dict(),
            "optim_discriminator": self.optimizerD.state_dict(),
            "optim_generator": self.optimizerG.state_dict(),
        }
        if self.config.pretrain_classifier: d["discriminator2"] = self.netD2.state_dict()
        torch.save(d, out_path)
        logging.info(f"Saved model (on epoch(?): {epoch_number}) to {out_path.resolve()}")

    def _create_network(self):
        # TODO: Rename model_name - perhaps something like "gan_type" would be fine. config.model_name already in use -> This could cause confusion.
        if self.model_name == "dcgan":  # Note that this is not self.config.model_name! This here is passed as args.
            logging.info(f"self.config.kernel_size: {self.config.kernel_size}")
            logging.info(f"self.config.image_size: {self.config.image_size}")

            from gan_compare.training.networks.dcgan.discriminator import (
                Discriminator
            )
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

            from gan_compare.training.networks.dcgan.generator import (
                Generator
            )
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
                leakiness=self.config.leakiness,
            ).to(self.device)

            self.netD = Discriminator(
                ndf=self.config.ndf,
                nc=self.config.nc,
                ngpu=self.config.ngpu,
                leakiness=self.config.leakiness,
                bias=False,
            ).to(self.device)

        else:
            raise ValueError(f"Unknown gan_type. Please revise the name of the GAN model: {self.model_name}")

        # Check if channel size is correct
        if self.config.conditional:
            assert (
                    self.config.nc == 2 or (self.config.nc == 4 and self.config.model_name == "swin_transformer")
            ), "To use conditional input, change number of channels (nc) to 2 (default) or 4 (swin transformer)."
        else:
            assert (
                    self.config.nc == 1 or (self.config.nc == 3 and self.config.model_name == "swin_transformer")
            ), "Without conditional input into GAN, change number of channels (nc) to 1 (default) or 3 (swin transformer)."

        # Handle multi-gpu if desired

        if (self.device.type == "cuda") and (self.config.ngpu > 1):
            self.netG = nn.DataParallel(self.netG, list(range(self.config.ngpu)))
        # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
        self.netG.apply(weights_init)

        # Print the generator model
        logging.info(self.netG)

        # Handle multi-gpu if desired
        if (self.device.type == "cuda") and (self.config.ngpu > 1):
            self.netD = nn.DataParallel(self.netD, list(range(self.config.ngpu)))

        # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
        self.netD.apply(weights_init)

        # Print the discriminator model
        logging.info(self.netD)

        if self.config.pretrain_classifier:
            if self.config.is_pretraining_adversarial:  # real/fake prediction
                num_classes = 1
            self.netD2 = get_classifier(self.config, num_classes=num_classes).to(self.device)
            if (self.device.type == "cuda") and (self.config.ngpu > 1):
                self.netD2 = nn.DataParallel(self.netD2, list(range(self.config.ngpu)))
            self.netD2.apply(weights_init)
            logging.info(self.netD2)

    def _netG_update(self, netD, optimizerG, fake_images, fake_conditions, epoch: int, are_outputs_logits: bool = False,
                     retain_graph: bool = False, is_G_updated: bool=True):
        ''' Update Generator network: maximize log(D(G(z))) '''

        # Generate label is repeated each time due to varying b_size i.e. last batch of epoch has less images
        # Here, the "real" label is needed, as the fake labels are "real" for generator cost.
        # label smoothing is False, as this option would decrease the loss of the generator.
        labels = torch.full((fake_images.size(0),), self._get_labels(smoothing=False).get('real'), dtype=torch.float,
                            device=self.device)

        # Since we just updated D, perform another forward pass of all-fake batch through the updated D.
        # The generator loss of the updated discriminator should be higher than the previous one.
        if self.config.conditional:
            output = netD(fake_images, fake_conditions).view(-1)
        else:
            output = netD(fake_images).view(-1)

        # Calculate G's loss based on this output
        errG = self._compute_loss(output=output, label=labels, epoch=epoch, are_outputs_logits=are_outputs_logits)

        # D output mean on the second generator input
        D_G_z2 = output.mean().item()

        if is_G_updated:
            # Calculate gradients for G
            errG.backward(
                retain_graph=retain_graph)  # another call to this backward() will happen if we pretrain the classifier
            # Update G
            optimizerG.step()

        return output, D_G_z2, errG

    def _netD_update(self, netD, optimizerD, real_images, fake_images, epoch: int, are_outputs_logits: bool = False,
                     real_conditions=None, fake_conditions=None):
        ''' Update Discriminator network on real AND fake data. '''

        # Forward pass real batch through D
        output_real, errD_real, D_x = self._netD_forward_backward_pass(
            netD,
            real_images,
            self._get_labels().get('real'),
            real_conditions,
            epoch=epoch,
            are_outputs_logits=are_outputs_logits,
        )
        # remove gradient only if fake_conditions is a torch tensor
        if fake_conditions is not None:
            fake_conditions = fake_conditions.detach()

        # Forward pass fake batch through D
        output_fake, errD_fake, D_G_z1 = self._netD_forward_backward_pass(
            netD,
            fake_images.detach(),
            self._get_labels().get('fake'),
            fake_conditions,
            epoch=epoch,
            are_outputs_logits=are_outputs_logits,
        )

        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake

        # Update D
        optimizerD.step()

        return output_real, errD_real, D_x, output_fake, errD_fake, D_G_z1, errD

    def _netD_forward_backward_pass(self, netD, images, label_as_float, conditions, epoch: int,
                                    are_outputs_logits: bool = False):
        ''' Forward and backward pass through discriminator network '''

        # Forward pass batch through D
        output = None
        if self.config.conditional:
            output = netD(images, conditions).view(-1)
        else:
            output = netD(images).view(-1)

        # Generate label is repeated each time due to varying b_size i.e. last batch of epoch has less images
        labels = torch.full((images.size(0),), label_as_float, dtype=torch.float, device=self.device)

        # Calculate loss on all-real batch
        errD = self._compute_loss(output=output, label=labels, epoch=epoch, are_outputs_logits=are_outputs_logits)

        # Calculate gradients for D in backward pass of real data batch
        errD.backward()
        D_input = output.mean().item()

        return output, errD, D_input

    def _compute_loss(self, output, label, epoch: int, are_outputs_logits: bool = False):
        ''' Setting the loss function. Computing and returning the loss. '''

        # Avoiding to reset self.criterion on each loss calculation
        is_output_type_changed = self.are_outputs_logits != are_outputs_logits
        self.are_outputs_logits = are_outputs_logits
        if not hasattr(self, 'criterion') or self.criterion is None or is_output_type_changed:
            # Initialize standard criterion. Note: Could be moved to config.
            if are_outputs_logits:
                self.criterion = nn.BCEWithLogitsLoss()
            else:
                self.criterion = nn.BCELoss()
        if self.config.switch_loss_each_epoch:
            # Note this design decision: switch_loss_each_epoch:bool=True overwrites use_lsgan_loss:bool=False
            if epoch % 2 == 0:
                # if epoch is even, we use BCE loss, if it is uneven, we use least square loss.
                logging.debug(
                    f'switch_loss={self.config.switch_loss_each_epoch}, epoch={epoch}, epoch%2 = {epoch % 2} == 0 -> BCE loss.')
                return self.criterion(output, label)
            else:
                logging.debug(
                    f'switch_loss={self.config.switch_loss_each_epoch}, epoch={epoch}, epoch%2 = {epoch % 2} != 0 -> LS loss.')
                if not are_outputs_logits:
                    return 0.5 * torch.mean((output - label) ** 2)
                else:
                    raise Exception("Please revise ls loss before using it with logits as input.")
        else:
            if self.model_name != "lsgan" and not self.config.use_lsgan_loss:
                # Standard criterion defined above - e.g. Vanilla GAN's binary cross entropy (BCE) loss
                return self.criterion(output, label)
            else:
                # Least Square Loss - https://arxiv.org/abs/1611.04076
                if not are_outputs_logits:
                    return 0.5 * torch.mean((output - label) ** 2)
                else:
                    raise Exception("Please revise ls loss before using it with logits as input.")

    def _get_labels(self, smoothing: bool = True):
        # if enabled, let's smooth the labels for "real" (--> real !=1)
        if self.config.use_one_sided_label_smoothing and smoothing:
            smoothed_real_label: float = random.uniform(self.config.label_smoothing_start,
                                                        self.config.label_smoothing_end)
            logging.debug(f"smoothed_real_label = {smoothed_real_label}")
            return {"real": smoothed_real_label, "fake": 0.0}
        return {"real": 1.0, "fake": 0.0}

    def _get_random_conditions(self, minimum=None, maximum=None, batch_size=None, requires_grad=False):
        if minimum is None:
            minimum = self.config.condition_min

        if maximum is None:
            # Need to add +1 here to allow torch.rand/randint to create samples with number = condition_max
            maximum = self.config.condition_max + 1

        if batch_size is None:
            # sometimes we might want to pass batch_size i.e. in the last batch of an epoch that might have less
            # training samples than previous batches.
            batch_size = self.config.batch_size

        if self.config.conditioned_on == 'density' and not self.config.is_condition_categorical and not self.config.is_condition_binary:
            # here we need a float randomly drawn from a set of possible values (0.0, 0.33, 0.67, 1.0) for breast density (1 - 4)
            conditions = []
            condition_value_options = list(DENSITY_DICT.values())
            for i in range(batch_size):
                # number out of [-1,1] multiplied by noise term parameter. Round for 2 digits
                noise = round(random.uniform(-1, 1) * self.config.added_noise_term, 2)
                # get condition with noise normalised between 0 and 1.
                condition_w_noise = max(min(random.choice(condition_value_options) + noise, 1.), 0.)
                conditions.append(condition_w_noise)
            condition_tensor = torch.tensor(conditions, device=self.device, requires_grad=requires_grad)
            logging.debug(f'random condition_tensor: {condition_tensor}')
            return condition_tensor
        else:
            # now we want an integer rather than a float.
            return torch.randint(
                minimum,
                maximum,
                (batch_size,),
                device=self.device,
                requires_grad=requires_grad)


    def handle_G_updates(self, iteration: int, fake_images, fake_conditions, epoch: int):

        self.netG.zero_grad()

        # return variable init
        output_fake_2 = None
        D_G_z2 = None
        errG = None
        output_fake_2_2 = None
        D2_G_z = None
        errG_2 = None

        if self.config.pretrain_classifier:
            # In case D2 only backpropagates after a certain number of epochs has passed.
            is_D2_backpropagated:bool = epoch >= self.config.start_backprop_D2_into_G_after_epoch
            if epoch == self.config.start_backprop_D2_into_G_after_epoch and iteration == 0:
                logging.info(f"As we have reached epoch={epoch}, we now start to backpropagate into G the gradients of D2 ({self.config.model_name}).")
            # Swin transformer returns last layer logits instead of probabilities
            are_outputs_logits = True if self.config.model_name == "swin_transformer" else False

        # Checking which D should backpropagate into G in this epoch.
        if self.config.pretrain_classifier and self.config.are_Ds_alternating_to_update_G and is_D2_backpropagated:

            # We always pass the two outputs of D1 and D2 through G, but only update with one of the outputs.
            if iteration % 2 == 0:
                # D1 output passed through G, AND backpropagated
                output_fake_2, D_G_z2, errG = self._netG_update(
                    netD=self.netD,
                    optimizerG=self.optimizerG,
                    fake_images=fake_images,
                    fake_conditions=fake_conditions,
                    retain_graph=False,
                    epoch=epoch,
                    are_outputs_logits=False,
                    is_G_updated = True
                )
                # D2 output passed through G, NOT backpropagated
                output_fake_2_2, D2_G_z, errG_2 = self._netG_update(
                    netD=self.netD2,
                    optimizerG=self.optimizerG,
                    fake_images=fake_images.detach(),
                    fake_conditions=fake_conditions,
                    retain_graph=False,
                    epoch=epoch,
                    are_outputs_logits=are_outputs_logits,
                    is_G_updated=False
                )
            else:
                # D1 output passed through G, NOT backpropagated
                output_fake_2, D_G_z2, errG = self._netG_update(
                    netD=self.netD,
                    optimizerG=self.optimizerG,
                    fake_images=fake_images,
                    fake_conditions=fake_conditions,
                    retain_graph=False,
                    epoch=epoch,
                    are_outputs_logits=False,
                    is_G_updated=False
                )
                # D2 output passed through G, AND backpropagated
                output_fake_2_2, D2_G_z, errG_2 = self._netG_update(
                    netD=self.netD2,
                    optimizerG=self.optimizerG,
                    fake_images=fake_images.detach(),
                    fake_conditions=fake_conditions,
                    retain_graph=False,
                    epoch=epoch,
                    are_outputs_logits=are_outputs_logits,
                    is_G_updated=True
                )
        else:
            output_fake_2, D_G_z2, errG = self._netG_update(
                netD=self.netD,
                optimizerG=self.optimizerG,
                fake_images=fake_images,
                fake_conditions=fake_conditions,
                retain_graph=self.config.pretrain_classifier,
                # another call to backward() will happen if we pretrain the classifier
                epoch=epoch,
                are_outputs_logits=False,
                is_G_updated=True,
            )

            if self.config.pretrain_classifier:
                self.netG.zero_grad()
                output_fake_2_2, D2_G_z, errG_2 = self._netG_update(
                    netD=self.netD2,
                    optimizerG=self.optimizerG,
                    fake_images=fake_images.detach(),
                    fake_conditions=fake_conditions,
                    retain_graph=False,
                    epoch=epoch,
                    are_outputs_logits=are_outputs_logits,
                    is_G_updated=is_D2_backpropagated,
                )
        return output_fake_2, D_G_z2, errG, output_fake_2_2, D2_G_z, errG_2

    def train(self):

        # As we train a new model (atm no continued checkpoint training), we create new model dir and save config.
        self._save_config()

        # Create batch of latent vectors that we will use to visualize the progression of the generator
        # For convenience, let's use the specified batch size = number of fixed noise random tensors
        fixed_noise = torch.randn(self.config.batch_size, self.config.nz, 1, 1, device=self.device)

        # Create batch of fixed conditions that we will use to visualize the progression of the generator
        fixed_condition = None
        if self.config.conditional:
            fixed_condition = self._get_random_conditions()

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

        running_loss_of_discriminator2 = None  # set to None for function calls later below
        running_loss_of_generator_D2 = None
        running_real_discriminator2_accuracy = None
        running_fake_discriminator2_accuracy = None
        D2_losses = None
        G2_losses = None

        if self.config.pretrain_classifier:
            # TODO: Check if the learning rate should differ between swin/transformers and cnn.
            self.optimizerD2 = optim.Adam(
                self.netD2.parameters(),
                lr=self.config.lr,
                betas=(self.config.beta1, 0.999),
                weight_decay=self.config.weight_decay,
            )
            running_loss_of_generator_D2 = 0.
            running_loss_of_discriminator2 = 0.
            running_real_discriminator2_accuracy = 0.
            running_fake_discriminator2_accuracy = 0.
            D2_losses = []
            G2_losses = []

        # Training Loop
        logging.info(f"Starting Training on {self.device}.. Image size: {self.config.image_size}")
        if self.config.conditional:
            logging.info(
                f"Training conditioned on: {self.config.conditioned_on}"
                f"{' as a continuous (with random noise: ' + f'{self.config.added_noise_term}' + ') and not' * (1 - self.config.is_condition_categorical)} as a categorical variable")
        # For each epoch
        for epoch in range(self.config.num_epochs):
            # We check if netD2 was initialized, which means self.confif.pretrain_classifier was true.
            if hasattr(self, 'netD2') and epoch >= self.config.start_training_D2_after_epoch:
                if not self.config.pretrain_classifier or epoch == 0:
                    logging.info(f"As we have reached epoch={epoch}, we now start training D2 ({self.config.model_name}).")
                self.config.pretrain_classifier = True
            else:
                # We only want to train D2 after a certain number of epochs, hence we set self.config.pretrain_classifier = False
                # until that number of epochs is reached.
                self.config.pretrain_classifier = False

            # For each batch in the dataloader
            for i, data in enumerate(self.dataloader, 0):

                # Unpack data (=image batch) alongside condition (i.e. birads number). Conditions are all -1 if unconditioned.
                try:
                    data, conditions, _, = data
                except:
                    data, conditions, _, _ = data

                # Format batch (fake and real), get images and, optionally, corresponding conditional GAN inputs
                real_images = data.to(self.device)

                # Cb_siompute the actual batch size (not from config!) for convenience
                b_size = real_images.size(0)
                logging.debug(f'b_size: {b_size}')
                logging.debug(f'condition: {conditions}')

                # Generate batch of latent vectors as input into generator to generate fake images
                noise = torch.randn(b_size, self.config.nz, 1, 1, device=self.device)

                real_conditions = None
                fake_conditions = None
                if self.config.conditional:
                    real_conditions = conditions.to(self.device)
                    # generate fake conditions
                    fake_conditions = self._get_random_conditions(batch_size=b_size)
                    logging.debug(f"fake_conditions: {fake_conditions}")
                    # Generate fake image batch with G (conditional_res64)
                    logging.debug(f"b_size: {b_size}")
                    logging.debug(f"noise.shape: {noise.shape}")
                    logging.debug(f"fake_conditions.shape: {fake_conditions.shape}")
                    fake_images = self.netG(noise, fake_conditions)
                else:
                    # Generate fake image batch with G (without condition)
                    fake_images = self.netG(noise)

                # We start by updating the discriminator
                # Reset the gradient of the discriminator of previous training iterations
                self.netD.zero_grad()

                # Perform a forward backward training step for D with optimizer weight update for real and fake data
                output_real, errD_real, D_x, output_fake_1, errD_fake, D_G_z1, errD = self._netD_update(
                    netD=self.netD,
                    optimizerD=self.optimizerD,
                    real_images=real_images,
                    fake_images=fake_images,
                    epoch=epoch,
                    are_outputs_logits=False,
                    real_conditions=real_conditions,
                    fake_conditions=fake_conditions,
                )

                if self.config.pretrain_classifier:
                    self.netD2.zero_grad()
                    are_outputs_logits = True if self.config.model_name == "swin_transformer" else False
                    output_real_2, errD2_real, D2_x, output_fake_1_2, errD2_fake, D2_G_z1, errD2 = self._netD_update(
                        netD=self.netD2,
                        optimizerD=self.optimizerD2,
                        real_images=real_images,
                        fake_images=fake_images.detach(),
                        epoch=epoch,
                        are_outputs_logits=are_outputs_logits,
                        real_conditions=real_conditions,
                        fake_conditions=fake_conditions,
                    )

                # After updating the discriminator, we now update the generator
                # Reset to zero as previous gradient should have already been used to update the generator network
                self.netG.zero_grad()

                # Perform a forward backward training step for G with optimizer weight update including a second
                # output prediction by D to get bigger gradients as D has been already updated on this fake image batch.
                output_fake_2, D_G_z2, errG, output_fake_2_2, D2_G_z, errG_2 = self.handle_G_updates(
                    iteration=i,
                    fake_images=fake_images,
                    fake_conditions=fake_conditions,
                    epoch=epoch
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
                # Update the running loss which is used in visualization
                running_loss_of_generator += errG.item()


                D_losses.append(errD.item())
                # Update the running loss which is used in visualization
                running_loss_of_discriminator += errD.item()

                if self.config.pretrain_classifier:
                    current_real_acc_2 = torch.sum(output_real_2 > self.config.discriminator_clf_threshold).item() / \
                                         list(output_real_2.size())[0]
                    running_real_discriminator2_accuracy += current_real_acc_2
                    current_fake_acc_2 = torch.sum(output_fake_1_2 < self.config.discriminator_clf_threshold).item() / \
                                         list(output_fake_1_2.size())[0]
                    running_fake_discriminator2_accuracy += current_fake_acc_2

                    D2_losses.append(errD2.item())
                    running_loss_of_discriminator2 += errD2.item()

                    G2_losses.append(errG_2.item())
                    running_loss_of_generator_D2 += errG_2.item()

                # Output training stats on each iteration length threshold
                if i % self.config.num_iterations_between_prints == 0:
                    if self.config.pretrain_classifier:
                        # While not necessarily backpropagating into G, both D1 and D2 are used and we have all possible numbers available.
                        logging.info(
                            '[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_D2: %.4f\tLoss_G_D1: %.4f\tLoss_G_D2: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f\tAcc(D(x)): %.4f\tAcc(D(G(z)): %.4f\tD2(x): %.4f\tD2(G(z)): %.4f / %.4f\tAcc(D2(x)): %.4f\tAcc(D2(G(z)): %.4f'
                            % (epoch, self.config.num_epochs - 1, i, len(self.dataloader),
                               errD.item(), errD2.item(), errG.item(), errG_2.item(), D_x, D_G_z1, D_G_z2,
                               current_real_acc,
                               current_fake_acc, D2_x, D2_G_z1, D2_G_z, current_real_acc_2, current_fake_acc_2))
                    else:
                        # We only log D1 and G statistics, as D2 was not used in GAN training.
                        logging.info(
                            '[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f\tAcc(D(x)): %.4f\tAcc(D(G(z)): %.4f'
                            % (epoch, self.config.num_epochs - 1, i, len(self.dataloader),
                               errD.item(), errG.item(), D_x, D_G_z1, D_G_z2, current_real_acc, current_fake_acc))

                    # Add loss scalars to tensorboard
                    visualization_utils.add_value_to_tensorboard_loss_diagram(epoch=epoch,
                                                                              iteration=i,
                                                                              running_loss_of_generator=running_loss_of_generator,
                                                                              running_loss_of_generator_D2=running_loss_of_generator_D2,
                                                                              running_loss_of_discriminator=running_loss_of_discriminator,
                                                                              running_loss_of_discriminator2=running_loss_of_discriminator2)
                    # Add accuracy scalars to tensorboard
                    visualization_utils.add_value_to_tensorboard_accuracy_diagram(epoch=epoch,
                                                                                  iteration=i,
                                                                                  running_real_discriminator_accuracy=running_real_discriminator_accuracy,
                                                                                  running_fake_discriminator_accuracy=running_fake_discriminator_accuracy,
                                                                                  running_real_discriminator2_accuracy=running_real_discriminator2_accuracy,
                                                                                  running_fake_discriminator2_accuracy=running_fake_discriminator2_accuracy)
                    # Reset the running losses and accuracies
                    running_loss_of_generator = 0.
                    running_loss_of_discriminator = 0.
                    running_real_discriminator_accuracy = 0.
                    running_fake_discriminator_accuracy = 0.

                    if self.config.pretrain_classifier:
                        running_loss_of_generator_D2 = 0.
                        running_loss_of_discriminator2 = 0.
                        running_real_discriminator2_accuracy = 0.
                        running_fake_discriminator2_accuracy = 0.

                    # Visually check how the generator is doing by saving G's output on fixed_noise
                    # if (iters % self.config.num_iterations_between_prints * 10 == 0) or (
                    #        (epoch == self.config.num_epochs - 1) and (i == len(self.dataloader) - 1)):
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

            visualization_utils.plot_losses(D_losses=D_losses, D2_losses=D2_losses, G_losses=G_losses)
            if (epoch % 20 == 0 and epoch >= 5000):
                # TODO: Handle storage of model on each x epochs via config variable
                # Save on each 20th epoch starting at epoch 50.
                # self._save_model(epoch)
                pass
        self._save_model()

    def generate(self, model_checkpoint_path: Path, fixed_noise=None, fixed_condition=None,
                 num_samples: int = 10, birads: int = None) -> list:

        self.optimizerG = optim.Adam(
            self.netG.parameters(), lr=self.config.lr, betas=(self.config.beta1, 0.999)
        )
        checkpoint = torch.load(model_checkpoint_path, self.device)
        self.netG.load_state_dict(checkpoint["generator"])
        self.optimizerG.load_state_dict(checkpoint["optim_generator"])
        self.netG.eval()

        img_list = []
        # for ind in tqdm(range(num_samples)):
        if fixed_noise is None:
            fixed_noise = torch.randn(num_samples, self.config.nz, 1, 1, device=self.device)
        if self.config.conditional:
            if fixed_condition is None:
                fixed_condition = self._get_random_conditions(batch_size=num_samples)
            elif isinstance(fixed_condition, int):
                fixed_condition = self._get_random_conditions(minimum=fixed_condition, maximum=fixed_condition + 1,
                                                              batch_size=num_samples)
            fake = self.netG(fixed_noise, fixed_condition).detach().cpu().numpy()
        else:
            fake = self.netG(fixed_noise).detach().cpu().numpy()
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
                fixed_condition = self._get_random_conditions(requires_grad=False)

            # Visualize the model architecture of the generator
            visualization_utils.generate_tensorboard_network_graph(
                neural_network=self.netG,
                network_input_1=fixed_noise,
                network_input_2=fixed_condition)
            return visualization_utils
