"""VGG11 encoder
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn


class VGG11Encoder(nn.Module):


    def __init__(self, in_channels: int = 3):
        """Initialize the VGG11Encoder model."""
        super().__init__()
        
        # Block 1: output channel 64, 1 conv
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2: output channel 128, 1 conv
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 3: output channel 256, 2 convs
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 4: output channel 512, 2 convs
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 5: output channel 512, 2 convs
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

        features_dict = {}
        
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        if return_features:
            features_dict["conv1"] = x
        x = self.pool1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        if return_features:
            features_dict["conv2"] = x
        x = self.pool2(x)
        
        # Block 3
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = self.relu3_1(x)
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = self.relu3_2(x)
        if return_features:
            features_dict["conv3"] = x
        x = self.pool3(x)
        
        # Block 4
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.relu4_1(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.relu4_2(x)
        if return_features:
            features_dict["conv4"] = x
        x = self.pool4(x)
        
        # Block 5
        x = self.conv5_1(x)
        x = self.bn5_1(x)
        x = self.relu5_1(x)
        x = self.conv5_2(x)
        x = self.bn5_2(x)
        x = self.relu5_2(x)
        if return_features:
            features_dict["conv5"] = x
        x = self.pool5(x)
        
        bottleneck = x
        
        if return_features:
            return bottleneck, features_dict
        return bottleneck