'''
Author: wy 1955416359@qq.com
Date: 2025-04-03 11:28:53
LastEditors: wy 1955416359@qq.com
LastEditTime: 2025-04-03 11:38:13
FilePath: /seg2025/segformer/nets/segnext/decoder.py
Description: 

'''
#%%

from turtle import forward
import torch
import torch.nn.functional as F
import torch.nn as nn

from .hamburger import HamBurger
from .bricks import SeprableConv2d, ConvRelu, ConvBNRelu, resize


class HamDecoder(nn.Module):
    '''SegNext'''
    def __init__(self, outChannels, config, enc_embed_dims=[32,64,460,256]):
        super().__init__()

        ham_channels = config['ham_channels']
        # ham_channels = config['ham_channels']

        self.squeeze = ConvRelu(sum(enc_embed_dims[1:]), ham_channels)
        self.ham_attn = HamBurger(ham_channels, config)
        self.align = ConvRelu(ham_channels, outChannels)
       
    def forward(self, features):
        
        features = features[1:] # drop stage 1 features b/c low level
        features = [resize(feature, size=features[-3].shape[2:], mode='bilinear') for feature in features]
        x = torch.cat(features, dim=1)

        x = self.squeeze(x)
        x = self.ham_attn(x)
        x = self.align(x)       

        return x


#%%

# import torch.nn.functional as F

# def resize(input,
#            size=None,
#            scale_factor=None,
#            mode='nearest',
#            align_corners=None,
#            warning=True):

#     return F.interpolate(input, size, scale_factor, mode, align_corners)

# inputs = [resize(
#         level,
#         size=x[0].shape[2:],
#         mode='bilinear',
#         align_corners=False
#     ) for level in x]

# for i in range(4):
#     print(x[i].shape)
# for i in range(4):
#     print(inputs[i].shape)



# inputs = torch.cat(inputs, dim=1)
# print(inputs.shape)
