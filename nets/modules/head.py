'''
Author: wy 1955416359@qq.com
Date: 2025-04-19 01:00:23
LastEditors: wy 1955416359@qq.com
LastEditTime: 2025-04-19 01:06:28
FilePath: /seg2025/segformer/nets/modules/head.py
Description: 

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import DepthwiseSeparableConv

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=256, atrous_rates=(6, 12, 18)):
        super(ASPP, self).__init__()
        # 1x1 convolution branch
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        # 3x3 convolution branches with different dilation rates
        # self.convs = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=rate, dilation=rate),
        #         nn.BatchNorm2d(mid_channels),
        #         nn.ReLU(inplace=True)
        #     ) for rate in atrous_rates
        # ])
        # Depthwise separable convolution branches with different dilation rates
        self.convs = nn.ModuleList([
            DepthwiseSeparableConv(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=rate,
                dilation=rate
            ) for rate in atrous_rates
        ])
        # Global average pooling branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        # Total channels after concatenation: mid_channels * (1 + len(atrous_rates) + 1)
        total_channels = mid_channels * (1 + len(atrous_rates) + 1)
        # Final 1x1 convolution to reduce channels
        self.final_conv = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.size()[2:]  # Get input spatial size
        # 1x1 convolution
        conv1 = self.conv1(x)
        # 3x3 convolutions
        convs = [conv(x) for conv in self.convs]
        # Global pooling
        global_feat = self.global_pool(x)
        global_feat = F.interpolate(global_feat, size=size, mode='bilinear', align_corners=True)
        # Concatenate all branches
        all_feats = [conv1] + convs + [global_feat]
        cat = torch.cat(all_feats, dim=1)
        # Final convolution
        out = self.final_conv(cat)
        return out