"""
Credits: richardosuala (Richard Osuala)
"""

## Pypi imports
import matplotlib


matplotlib.use('agg')
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
import torch



class VisualizationUtils:
    """This class contains different visualization mechanisms for the GAN training process such as tensorboard writers."""

    def __init__(
        self,
        num_iterations_per_epoch: int,
        num_iterations_between_prints: int,
        tensorboard_writer_dir: str = "../runs",
    ):
        """
        :param tensorboard_writer_dir: The location in the fs where tensorboard visualization files are stored.
        """
        # setup tensorboard writer
        self.tensorboard_writer = SummaryWriter(tensorboard_writer_dir)
        self.num_iterations_per_epoch = num_iterations_per_epoch
        self.num_iterations_between_prints = num_iterations_between_prints


    def generate_tensorboard_network_graph(
        self, neural_network, network_input_1, network_input_2
    ):
        # Add the model architecture to tensorboard
        self.tensorboard_writer.add_graph(
            model=neural_network, input_to_model=(network_input_1, network_input_2)
        )
        self.tensorboard_writer.close()

    def add_value_to_tensorboard_accuracy_diagram(
        self,
        epoch,
        iteration,
        running_real_discriminator_accuracy,
        running_fake_discriminator_accuracy,
        main_tag: str = "discriminator accuracy",
    ):
        # Number of the current iteration in the global training process
        current_global_iteration: int = (
            epoch * self.num_iterations_per_epoch + iteration
        )

        # Initialize a dictionary that holds both D's fake and real accuracy
        accuracy_dictionary = {
            "real data": running_real_discriminator_accuracy
            / self.num_iterations_between_prints,
            "fake data": running_fake_discriminator_accuracy
            / self.num_iterations_between_prints,
        }

        self.tensorboard_writer.add_scalars(
            main_tag=main_tag,
            tag_scalar_dict=accuracy_dictionary,
            global_step=current_global_iteration,
        )
        self.tensorboard_writer.close()

    def add_value_to_tensorboard_loss_diagram(
        self,
        epoch,
        iteration,
        running_loss_of_generator,
        running_loss_of_discriminator,
        main_tag: str = "training loss",
    ):
        # Number of the current iteration in the global training process
        current_global_iteration: int = (
            epoch * self.num_iterations_per_epoch + iteration
        )

        # Initialize a dictionary that holds both generator and discriminator losses
        loss_dictionary = {
            "generator": running_loss_of_generator / self.num_iterations_between_prints,
            "discriminator": running_loss_of_discriminator
            / self.num_iterations_between_prints,
        }

        # Write the loss_dictionary to the tensorboard
        self.tensorboard_writer.add_scalars(
            main_tag=main_tag,
            tag_scalar_dict=loss_dictionary,
            global_step=current_global_iteration,
        )
        self.tensorboard_writer.close()

    def add_generated_batch_to_tensorboard(
        self,
        neural_network,
        network_input_1,
        network_input_2,
        padding: int = 2,
        normalize: bool = True,
        img_name: str = "GAN-generated-img",
    ):
        with torch.no_grad():
            generated_image = (
                neural_network(network_input_1, network_input_2).detach().cpu()
            )
            img_grid = vutils.make_grid(
                generated_image, padding=padding, normalize=normalize
            )
            self.tensorboard_writer.add_image(img_name, img_grid)

    def plot_losses(
        self,
        G_losses,
        D_losses,
        figsize=(10, 5),
        title: str = "gan-loss-over-iterations",
        is_shown: bool = False,
        is_safed: bool = True,
    ):
        plt.figure(figsize=figsize)
        plt.title(title)
        plt.plot(G_losses, label="G losses")
        plt.plot(D_losses, label="D losses")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        if is_shown:
            plt.show()
        if is_safed:
            plt.savefig(title + ".jpg")
