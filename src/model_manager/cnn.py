from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch


class LeNet(nn.Module):
    def __init__(self, nc, nh, hw, num_classes, seed=1):
        torch.manual_seed(seed)
        super(LeNet, self).__init__()
        input_shape = (nc, nh, hw)

        self.max_pool = nn.MaxPool2d((2, 2))
        self.conv1 = nn.Conv2d(nc, 64, 5)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.flat_shape = self.get_flat_shape(input_shape)

        # self.max_pool = nn.DataParallel(self.max_pool)
        # self.conv1 = nn.DataParallel(self.conv1)
        # self.conv2 = nn.DataParallel(self.conv2)

        self.fc1 = nn.Linear(self.flat_shape, 1024)
        # self.fc1 = nn.DataParallel(self.fc1)

        self.fc2 = nn.Linear(1024, num_classes)
        # self.fc2 = nn.DataParallel(self.fc2)

    # noinspection PyArgumentList,PyTypeChecker
    def get_flat_shape(self, input_shape):
        dummy = Variable(torch.zeros(1, *input_shape))
        dummy = self.max_pool(self.conv1(dummy))
        dummy = self.max_pool(self.conv2(dummy))
        return dummy.data.view(1, -1).size(1)

    def forward(self, x_in):
        # conv 1
        x = self.conv1(x_in)
        x = self.max_pool(x)
        x = F.relu(x)
        # conv 2
        x = self.conv2(x)
        x = self.max_pool(x)
        x = F.relu(x)
        # flatten
        x = x.view(-1, self.flat_shape)
        # fc 1
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        # fc 2
        z = self.fc2(x)
        return z