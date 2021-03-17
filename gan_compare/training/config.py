from time import time

# Birads range
birads_min = 2
birads_max = 6

# l2 regularization in discriminator
weight_decay = 0.5

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 4

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Whether to use least square loss
use_lsgan_loss = False

# Leakiness for ReLUs
leakiness = 0.3

# Number of channels in the training images. For color images this is 3
nc = 2

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 10

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

output_model_dir = f"model_checkpoints/training_{time()}/{image_size}/"

pretrained_model_path = f"model_checkpoints/{image_size}/model.pt"
