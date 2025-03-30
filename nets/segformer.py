# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5

from .modules.block import C2PSA

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x
    
class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, num_classes=20, in_channels=[32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1):
        super(SegFormerHead, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            c1=embedding_dim*4,
            c2=embedding_dim,
            k=1,
        )

        self.linear_pred    = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout        = nn.Dropout2d(dropout_ratio)
    
    def forward(self, inputs):
        c1, c2, c3, c4 = inputs

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape
        
        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x


class SegFormerHeadWithC2PSA(nn.Module):
    """
    SegFormer with C2PSA: Enhanced Semantic Segmentation by Integrating C2PSA Module
    """
    def __init__(self, num_classes=20, in_channels=[32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1):
        super(SegFormerHeadWithC2PSA, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        # MLP 调整每个特征图的通道数
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        # C2PSA 模块，输入为拼接后的 embedding_dim * 4，输出保持一致
        self.c2psa = C2PSA(c1=embedding_dim * 4, c2=embedding_dim * 4, n=1, e=0.5)

        # 融合特征图的卷积模块
        self.linear_fuse = ConvModule(
            c1=embedding_dim * 4,
            c2=embedding_dim,
            k=1,
        )

        # 最终预测层
        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_ratio)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs

        # 获取 c4 的形状用于后续上采样
        n, _, h, w = c4.shape

        # 对每个特征图进行 MLP 处理并上采样到 c1 的大小
        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        # 拼接多层次特征图
        fused = torch.cat([_c4, _c3, _c2, _c1], dim=1)

        # 应用 C2PSA 增强特征表达
        fused = self.c2psa(fused)

        # 融合特征并生成分割结果
        _c = self.linear_fuse(fused)
        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x

# class SegFormer(nn.Module):
#     def __init__(self, num_classes = 21, phi = 'b0', pretrained = False):
#         super(SegFormer, self).__init__()
#         self.in_channels = {
#             'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512], 'b2': [64, 128, 320, 512],
#             'b3': [64, 128, 320, 512], 'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
#         }[phi]
#         self.backbone   = {
#             'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,
#             'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5,
#         }[phi](pretrained)
#         self.embedding_dim   = {
#             'b0': 256, 'b1': 256, 'b2': 768,
#             'b3': 768, 'b4': 768, 'b5': 768,
#         }[phi]
#         self.decode_head = SegFormerHead(num_classes, self.in_channels, self.embedding_dim)

#     def forward(self, inputs):
#         H, W = inputs.size(2), inputs.size(3)
        
#         x = self.backbone.forward(inputs)
#         x = self.decode_head.forward(x)
        
#         x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
#         return x

class SegFormer(nn.Module):
    def __init__(self, num_classes = 21, phi = 'b0', pretrained = False):
        super(SegFormer, self).__init__()
        self.in_channels = {
            'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512], 'b2': [64, 128, 320, 512],
            'b3': [64, 128, 320, 512], 'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
        }[phi]
        self.backbone   = {
            'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,
            'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5,
        }[phi](pretrained)
        self.embedding_dim   = {
            'b0': 256, 'b1': 256, 'b2': 768,
            'b3': 768, 'b4': 768, 'b5': 768,
        }[phi]
        # self.decode_head = SegFormerHeadWithC2PSA(num_classes, self.in_channels, self.embedding_dim)
        self.decode_head = SegFormerHead(num_classes, self.in_channels, self.embedding_dim)
        self.c2psa_layers = nn.ModuleList([
            None,  # c1
            None,  # c2 不使用 C2PSA
            C2PSA(self.in_channels[2], self.in_channels[2], n=3, e=0.5),  # c3
            None   # c4 不使用 C2PSA
        ])
    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        
        features = self.backbone.forward(inputs)
        # print(f"Backbone output shapes: {[f.shape for f in features]}")
        # x = self.decode_head.forward(x)
        # 对每个特征图应用 C2PSA
        # print(f"Enhanced features shapes: {[f.shape for f in enhanced_features]}")

        enhanced_features = []
        for i, (c2psa, feat) in enumerate(zip(self.c2psa_layers, features)):
            if c2psa is not None:  # 如果有 C2PSA 模块，则应用
                enhanced_features.append(c2psa(feat))
            else:  # 否则直接使用原始特征图
                enhanced_features.append(feat)

        # 传递给解码头
        x = self.decode_head(enhanced_features)
        
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x
    




