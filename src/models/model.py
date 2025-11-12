import math

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

from .layers.blocks import SimpleBlock, BatchBlock, DropoutBlock


class TableModel(nn.Module):
    def __init__(self,
                 img_size=(3, 512, 512),
                 wf=5,
                 depth=3,
                 conv_blocks_type='simple'):
        super().__init__()
        assert conv_blocks_type in ['simple', 'dropout', 'batch'], conv_blocks_type
        
        prev_channels = img_size[0]
        
        self.batch_norm = nn.BatchNorm2d(3)
        
        self.conv_blocks = nn.ModuleList()
        if conv_blocks_type == 'simple':
            for i in range(depth):
                self.conv_blocks.append(SimpleBlock(prev_channels,
                                                    2**(wf + i)))
                prev_channels = 2**(wf + i)
        if conv_blocks_type == 'batch':
            for i in range(depth):
                self.conv_blocks.append(BatchBlock(prev_channels, 2**(wf + i)))
                prev_channels = 2**(wf + i)
        if conv_blocks_type == 'dropout':
            for i in range(depth):
                self.conv_blocks.append(DropoutBlock(prev_channels,
                                                     2**(wf + i)))
                prev_channels = 2**(wf + i)
        
        out_param = self.calculate_count_param(img_size, depth, prev_channels)
        # self.head_batch_norm1 = nn.BatchNorm1d(out_param)
        self.head_linear = nn.Linear(out_param, 2)
        # self.head_batch_norm2 = nn.BatchNorm1d(out_param // 2)
        # self.head_linear2 = nn.Linear(out_param // 2, 2)
        self._init_weights()
        
    def calculate_count_param(self, img_size, depth, last_channels):
        out_r = int(img_size[1] / 2**depth)
        out_c = int(img_size[2] / 2**depth)
        return out_r * out_c * last_channels

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {
                nn.Linear,
                nn.Conv2d,
            }:
                nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode='fan_out', nonlinearity='relu',
                )
                if m.bias is not None:
                    fan_in, fan_out = \
                        nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)
    
    def forward(self, x):
        x = self.batch_norm(x)
        for i, conv in enumerate(self.conv_blocks):
            x = conv(x)
        
        conv_flat = torch.flatten(x, 1)
        # batch_norm_output = self.head_batch_norm1(conv_flat)
        out = F.dropout2d(conv_flat, 0.3)
        linear_output = self.head_linear(out)
        # linear_output = self.head_batch_norm2(linear_output)
        # linear_output = self.head_linear2(linear_output)
        
        return linear_output, torch.softmax(linear_output, 1)


class MobileNetWrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        
        self.mobile_net = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
        
        for param in self.mobile_net.parameters():
            param.requires_grad = False
        
        self.mobile_net.classifier[1] = \
            nn.Linear(self.mobile_net.last_channel, 2)
    
    def forward(self, x):
        out = self.mobile_net(x)
        return out, torch.softmax(out, 1)
