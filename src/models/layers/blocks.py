import torch
from torch import nn
import torch.nn.functional as F


class SimpleBlock(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            conv_channels,
            kernel_size=3,
            padding=1
        )
        self.conv2 = nn.Conv2d(
            conv_channels,
            conv_channels,
            kernel_size=3,
            padding=1
        )
    
    def forward(self, input_batch):
        out = torch.relu(self.conv1(input_batch))
        out = torch.relu(self.conv2(out))
        return F.max_pool2d(out, 2)
    

class BatchBlock(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            conv_channels,
            kernel_size=3,
            padding=1
        )
        # self.batch_norm1 = nn.BatchNorm2d(conv_channels)
        
        self.conv2 = nn.Conv2d(
            conv_channels,
            conv_channels,
            kernel_size=3,
            padding=1
        )
        self.batch_norm2 = nn.BatchNorm2d(conv_channels)
    
    def forward(self, input_batch):
        out = torch.relu(self.conv1(input_batch))
        out = torch.relu(self.batch_norm2(self.conv2(out)))
        return F.max_pool2d(out, 2)
    

class DropoutBlock(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            conv_channels,
            kernel_size=3,
            padding=1
        )
        self.conv2 = nn.Conv2d(
            conv_channels,
            conv_channels,
            kernel_size=3,
            padding=1
        )
    
    def forward(self, input_batch):
        out = torch.relu(self.conv1(input_batch))
        out = torch.relu(self.conv2(out))
        out = F.dropout2d(out, 0.3)
        return F.max_pool2d(out, 2)