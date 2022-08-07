import torch
import torch.nn as nn
import math
from models.quant_layer import *


class VGG(nn.Module):
    def __init__(self, num_classes=10, w_bits=4, a_bits=4):
        super(VGG, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            QuantConv2d(128, 128, 3, 1, 1, w_bits=w_bits, a_bits=a_bits),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            QuantConv2d(128, 256, 3, 1, 1, w_bits=w_bits, a_bits=a_bits),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            QuantConv2d(256, 256, 3, 1, 1, w_bits=w_bits, a_bits=a_bits),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            QuantConv2d(256, 512, 3, 1, 1, w_bits=w_bits, a_bits=a_bits),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            QuantConv2d(512, 512, 3, 1, 1, w_bits=w_bits, a_bits=a_bits),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )
        self.fc = nn.Linear(512*4*4, num_classes)

    def forward(self, x):
        fea = self.body(x)
        fea = fea.view(fea.size(0), -1)
        out = self.fc(fea)

        return out