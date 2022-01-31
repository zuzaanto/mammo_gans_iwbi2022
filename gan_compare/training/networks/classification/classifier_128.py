import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F

DROPOUT_RATE = 0.2


class Net(nn.Module):
    def __init__(self, num_labels: int):
        super(Net, self).__init__()
        self.num_labels = num_labels
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.bn4 = nn.BatchNorm2d(128)

        # 2 fully connected layers to transform the output of the convolution layers to the final output
        self.fc1 = nn.Linear(in_features=8 * 8 * 128, out_features=128)
        self.fcbn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(in_features=128, out_features=self.num_labels)
        self.dropout_rate = DROPOUT_RATE

    def forward(self, s):
        # apply the convolution layers, followed by batch normalisation,
        # maxpool and relu x 4
        s = self.bn1(self.conv1(s))  # batch_size x 16 x 128 x 128
        s = F.relu(F.max_pool2d(s, 2))  # batch_size x 16 x 64 x 64
        s = self.bn2(self.conv2(s))  # batch_size x 32 x 64 x 64
        s = F.relu(F.max_pool2d(s, 2))  # batch_size x 32 x 32 x 32
        s = self.bn3(self.conv3(s))  # batch_size x 64 x 32 x 32
        s = F.relu(F.max_pool2d(s, 2))  # batch_size x 64 x 16 x 16
        s = self.bn4(self.conv4(s))  # batch_size x 128 x 16 x 16
        s = F.relu(F.max_pool2d(s, 2))  # batch_size x 128 x 8 x 8
        # flatten the output for each image
        s = s.view(-1, 8 * 8 * 128)  # batch_size x 8*8*128

        # apply 2 fully connected layers with dropout
        s = F.dropout(
            F.relu(self.fcbn1(self.fc1(s))), p=self.dropout_rate, training=self.training
        )  # batch_size x 128
        s = self.fc2(s)  # batch_size x num_labels

        logits = F.log_softmax(s, dim=1)

        return torch.exp(logits[:,-1]) # return only the probability of the true class
