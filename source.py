"""
EECS 445 - Introduction to Machine Learning
Winter 2021 - Project 2
Source CNN
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.source import Source
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from utils import config


class Source(nn.Module):
    def __init__(self):
        super().__init__()

        ## TODO: define each layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5,5), stride=(2,2), padding=2)
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(5,5), stride=(2,2), padding=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=8, kernel_size=(5,5), stride=(2,2), padding=2)
        self.fc1 = nn.Linear(32, 8)
        ##

        self.init_weights()

    def init_weights(self):
        torch.manual_seed(42)
        for conv in [self.conv1, self.conv2, self.conv3]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5 * 5 * C_in))
            nn.init.constant_(conv.bias, 0.0)

        ## TODO: initialize the parameters for [self.fc1]
        C_in1 = self.fc1.weight.size(1)
        nn.init.normal_(self.fc1.weight, 0.0, 1 / sqrt(C_in1))
        nn.init.constant_(self.fc1.bias, 0.0)
        ##

    def forward(self, x):
        N, C, H, W = x.shape

        ## TODO: forward pass
        x = F.relu(self.conv1(x))
        #pooling layer
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        #pooling layer
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = x.view(-1,32)
        x = self.fc1(x)
        ##

        return x


