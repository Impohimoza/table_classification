import math

import torch
from torch import nn

from .layers.blocks import SimpleBlock


class TableModel(nn.Module):
    def __init__(self,
                 img_size=(3, 512, 512),
                 wf=7,
                 depth=3,
                 conv_blocks_type='simple'):
        super().__init__()
        assert conv_blocks_type in ['simple'], conv_blocks_type
        
        prev_channels = img_size[0]
        
        self.batch_norm = nn.BatchNorm2d(3)
        
        self.conv_blocks = nn.ModuleList()
        for i in range(depth):
            self.conv_blocks.append(SimpleBlock(prev_channels, 2**(wf - i)))
            prev_channels = 2**(wf - i)
        
        out_param = self.calculate_count_param(img_size, depth, prev_channels)
        self.head_batch_norm = nn.BatchNorm1d(out_param)
        self.head_linear = nn.Linear(out_param, 2)
        self._init_weights()
        
    def calculate_count_param(self, img_size, depth, last_channels):
        out_size = (img_size[1] * img_size[2] * last_channels)
        return int(out_size / 2**(2 * depth))

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
        batch_norm_output = self.head_batch_norm(conv_flat)
        linear_output = self.head_linear(batch_norm_output)
        
        return linear_output, torch.softmax(linear_output, 1)
        