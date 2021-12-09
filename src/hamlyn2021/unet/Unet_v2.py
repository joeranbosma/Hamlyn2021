# coding: utf-8

"""
Module to create a UNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
import numpy as np
from torchvision import datasets, models, transforms
import cv2 
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
import os
import time

# define U-net
def double_conv(in_channels, out_channels, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class UNet(nn.Module):

    def __init__(self, num_channels=3):
        super().__init__()
        
        self.dconv_down1 = double_conv(num_channels, 32)
        self.dconv_down2 = double_conv(32, 64)
        self.dconv_down3 = double_conv(64, 128)
        self.dconv_down4 = double_conv(128, 256)
        self.dconv_down5 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.dropout = nn.Dropout2d(0.5)
        self.dconv_up4 = double_conv(256 + 512, 256)
        self.dconv_up3 = double_conv(128 + 256, 128)
        self.dconv_up2 = double_conv(128 + 64, 64)
        self.dconv_up1 = double_conv(64 + 32, 32)

        self.conv_last = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        
        #######   ENCODER ###############
        
        conv1 = self.dconv_down1(x)  # FOV: 5x5
        conv1 = self.dropout(conv1)
        x = self.maxpool(conv1)  # FOV: 10x10

        # implement encoder layers conv2, conv3 and conv4
        
        conv2 = self.dconv_down2(x)  # FOV: 12x12
        conv2 = self.dropout(conv2)
        x = self.maxpool(conv2)  # FOV: 24x24

        conv3 = self.dconv_down3(x)  # FOV: 26x26
        conv3 = self.dropout(conv3)
        x = self.maxpool(conv3)  # FOV: 52x52

        conv4 = self.dconv_down4(x)  # FOV: 54x54
        conv4 = self.dropout(conv4)
        x = self.maxpool(conv4)  # FOV: 108x108

        # implement bottleneck
        
        conv5 = self.dconv_down5(x)  # FOV: 110x110
        conv5 = self.dropout(conv5)
        # ---------------------------------------------------------------------------------------------------------------------
       
        #######   DECODER ###############
        
        # Implement the decoding layers
        
        deconv4 = self.upsample(conv5)
        deconv4 = torch.cat([deconv4, conv4], dim=1)  
        deconv4  = self.dconv_up4(deconv4)
        deconv4 = self.dropout(deconv4)

        deconv3 = self.upsample(deconv4 )       
        deconv3 = torch.cat([deconv3, conv3], dim=1)
        deconv3 = self.dconv_up3(deconv3)
        deconv3 = self.dropout(deconv3)

        deconv2 = self.upsample(deconv3)      
        deconv2 = torch.cat([deconv2, conv2], dim=1)
        deconv2 = self.dconv_up2(deconv2)
        deconv2 = self.dropout(deconv2)
       
        deconv1 = self.upsample(deconv2)   
        deconv1 = torch.cat([deconv1, conv1], dim=1)
        deconv1 = self.dconv_up1(deconv1)
        deconv1 = self.dropout(deconv1)

        #---------------------------------------------------------------------------------------------------------------------
        #out = F.sigmoid(self.conv_last(deconv1))
        deconv1 = self.conv_last(deconv1)
        out = torch.sigmoid(deconv1)
        
        return out

class UNet_7down(nn.Module):

    def __init__(self, num_channels=3):
        super().__init__()
        
        self.dconv_down1 = double_conv(num_channels, 32)
        self.dconv_down2 = double_conv(32, 64)
        self.dconv_down3 = double_conv(64, 128)
        self.dconv_down4 = double_conv(128, 256)
        self.dconv_down5 = double_conv(256, 320)
        self.dconv_down6 = double_conv(320, 320)
        self.dconv_down7 = double_conv(320, 320)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.dropout = nn.Dropout2d(0.5)
        self.dconv_up6 = double_conv(320 + 320, 320)
        self.dconv_up5 = double_conv(320 + 320, 320)
        self.dconv_up4 = double_conv(256 + 320, 256)
        self.dconv_up3 = double_conv(128 + 256, 128)
        self.dconv_up2 = double_conv(128 + 64, 64)
        self.dconv_up1 = double_conv(64 + 32, 32)

        self.conv_last = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        
        #######   ENCODER ###############
        
        conv1 = self.dconv_down1(x)  # FOV: 5x5
        conv1 = self.dropout(conv1)
        x = self.maxpool(conv1)  # FOV: 10x10

        # implement encoder layers conv2, conv3 and conv4
        
        conv2 = self.dconv_down2(x)  # FOV: 12x12
        conv2 = self.dropout(conv2)
        x = self.maxpool(conv2)  # FOV: 24x24

        conv3 = self.dconv_down3(x)  # FOV: 26x26
        conv3 = self.dropout(conv3)
        x = self.maxpool(conv3)  # FOV: 52x52

        conv4 = self.dconv_down4(x)  # FOV: 54x54
        conv4 = self.dropout(conv4)
        x = self.maxpool(conv4)  # FOV: 108x108

        conv5 = self.dconv_down5(x)  # FOV: 110x110
        conv5 = self.dropout(conv5)
        x = self.maxpool(conv5)  # FOV: 220x220

        conv6 = self.dconv_down6(x)  # FOV: 222x222
        conv6 = self.dropout(conv6)
        x = self.maxpool(conv6)  # FOV: 444x444

        # implement bottleneck
        
        conv7 = self.dconv_down7(x)  # FOV: 446x446
        conv7 = self.dropout(conv7)
        # ---------------------------------------------------------------------------------------------------------------------
       
        #######   DECODER ###############
        
        # Implement the decoding layers
        
        deconv6 = self.upsample(conv7)
        deconv6 = torch.cat([deconv6, conv6], dim=1)  
        deconv6  = self.dconv_up4(deconv6)
        deconv6 = self.dropout(deconv6)
        
        deconv5 = self.upsample(conv6)
        deconv5 = torch.cat([deconv5, conv5], dim=1)  
        deconv5  = self.dconv_up4(deconv5)
        deconv5 = self.dropout(deconv5)
        
        deconv4 = self.upsample(conv5)
        deconv4 = torch.cat([deconv4, conv4], dim=1)  
        deconv4  = self.dconv_up4(deconv4)
        deconv4 = self.dropout(deconv4)

        deconv3 = self.upsample(deconv4 )       
        deconv3 = torch.cat([deconv3, conv3], dim=1)
        deconv3 = self.dconv_up3(deconv3)
        deconv3 = self.dropout(deconv3)

        deconv2 = self.upsample(deconv3)      
        deconv2 = torch.cat([deconv2, conv2], dim=1)
        deconv2 = self.dconv_up2(deconv2)
        deconv2 = self.dropout(deconv2)
       
        deconv1 = self.upsample(deconv2)   
        deconv1 = torch.cat([deconv1, conv1], dim=1)
        deconv1 = self.dconv_up1(deconv1)
        deconv1 = self.dropout(deconv1)

        #---------------------------------------------------------------------------------------------------------------------
        #out = F.sigmoid(self.conv_last(deconv1))
        deconv1 = self.conv_last(deconv1)
        out = torch.sigmoid(deconv1)
        
        return out
