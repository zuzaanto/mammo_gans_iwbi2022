import torch
import torch.nn.parallel
from torch import autograd
from torch import nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


MODE = 'wgan-gp'  # Valid options are dcgan, wgan, or wgan-gp
DIM = 128  # This overfits substantially; you're probably better off with 64
LAMBDA = 10  # Gradient penalty lambda hyperparameter
CRITIC_ITERS = 5  # How many critic iterations per generator iteration
BATCH_SIZE = 64  # Batch size
ITERS = 200000  # How many generator iterations to train for
OUTPUT_DIM = 3072  # Number of pixels in CIFAR10 (3*32*32)



def compute_gradient_penalty(netD, real_images, fake_images, wgangp_lambda, device: str="cpu"):
    ''' gradient penalty computation according to paper https://arxiv.org/pdf/1704.00028.pdf

    Adapted from https://github.com/caogang/wgan-gp/blob/ae47a185ed2e938c39cf3eb2f06b32dc1b6a2064/gan_cifar10.py#L74
    '''

    # Determine image shape
    b_size = real_images.size()[0]

    # Define alpha, which is a random number a ∼ U[0, 1].
    alpha = torch.rand(b_size, 1)
    alpha = alpha.expand_as(real_images)
    alpha = alpha.to(device)

    # Compute the x_hat as a random spread between real and generated
    x_hat = alpha * real_images + ((1 - alpha) * fake_images)
    x_hat.to(device)
    x_hat = autograd.Variable(x_hat, requires_grad=True)

    # Pass x_hat as a random spread between real and generated
    disc_x_hat = netD(x_hat)

    # Compute gradient for disc_x_hat
    gradients = autograd.grad(outputs=disc_x_hat,
                              inputs=x_hat,
                              grad_outputs=torch.ones(disc_x_hat.size()).to(device),
                              create_graph=True,
                              retain_graph=True,
                              only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    # Compute the final gradient penalty where _lambda is the gradient penalty coefficient
    gradient_penalty = wgangp_lambda * (((gradients.norm(2, dim=1) - 1) ** 2).mean())

    return gradient_penalty


def compute_wgangp_loss(d_fake_images, d_real_images, gradient_penalty):
    d_loss = d_fake_images.mean() - d_real_images.mean() + gradient_penalty
    g_loss = - d_fake_images.mean()
    return d_loss, g_loss


def compute_ls_loss(output, label):
    return 0.5 * torch.mean((output - label) ** 2)

