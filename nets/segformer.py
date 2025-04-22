# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import imp
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5

from .modules.block import C2PSA
from .modules.conv import RepConv
from .modules.block import SCDown
from .modules.scsa import SCSA
from .modules.block import SCSABlock
from .modules.block import Bottleneck
from .modules.head import ASPP
from .modules.conv import HWD

# class MLP(nn.Module):
#     """
#     Linear Embedding
#     """
#     def __init__(self, input_dim=2048, embed_dim=768):
#         super().__init__()
#         self.proj = nn.Linear(input_dim, embed_dim)

#     def forward(self, x):
#         x = x.flatten(2).transpose(1, 2)
#         x = self.proj(x)
#         return x
    
class MLP(nn.Module):
    """
    Hybrid CNN-MLP for enhanced feature embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, input_dim // 2, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(input_dim // 2)
        self.act = nn.ReLU()
        self.proj = nn.Linear(input_dim // 2, embed_dim)

    def forward(self, x):
        x = self.conv(x)  # [B, C, H, W] -> [B, C//2, H, W]
        x = self.norm(x)
        x = self.act(x)
        x = x.flatten(2).transpose(1, 2)  # [B, C//2, H, W] -> [B, H*W, C//2]
        x = self.proj(x)  # [B, H*W, embed_dim]
        return x 
    


# class MLP(nn.Module):
#     """
#     MLP with Self-Attention for SegFormer
#     """
#     def __init__(self, input_dim=2048, embed_dim=768, num_heads=8):
#         super().__init__()
#         self.proj = nn.Linear(input_dim, embed_dim)
#         self.attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.norm2 = nn.LayerNorm(embed_dim)
#         self.ffn = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim * 2),
#             nn.GELU(),
#             nn.Linear(embed_dim * 2, embed_dim)
#         )

#     def forward(self, x):
#         x = x.flatten(2).transpose(1, 2)  # [B, C, H, W] -> [B, H*W, C]
#         x = self.proj(x)  # [B, H*W, embed_dim]
#         x = self.norm1(x)
#         attn_output, _ = self.attn(x, x, x)
#         x = x + attn_output
#         x = self.norm2(x)
#         x = x + self.ffn(x)
#         return x


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
        # self.down_c1 = SCDown(c1=embedding_dim, c2=embedding_dim, k=3, s=4)
        self.linear_fuse = ConvModule(
            c1=embedding_dim*4,
            c2=embedding_dim,
            k=1,
        )
        
        # self.linear_fuse = RepConv(c1=embedding_dim*4, c2=embedding_dim, k=3, s=1, p=0, act=True)

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
        # _c1 = self.down_c1(_c1)
        # _c1 = F.interpolate(_c1, size=c1.size()[2:], mode='bilinear', align_corners=False)
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x
    




class UPerHead(nn.Module):
    """
    UPerNet-inspired Decoder Head for SegFormer
    """
    def __init__(self, num_classes=20, in_channels=[32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1, pool_scales=(1, 2, 3, 6)):
        super(UPerHead, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        # MLP 投影层（与原 SegFormerHead 一致）
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        # FPN 部分：对每个特征层进行卷积处理
        self.fpn_convs = nn.ModuleList([
            ConvModule(c1=embedding_dim, c2=embedding_dim, k=3, p=1, act='relu'),
            ConvModule(c1=embedding_dim, c2=embedding_dim, k=3, p=1, act='relu'),
            ConvModule(c1=embedding_dim, c2=embedding_dim, k=3, p=1, act='relu'),
            ConvModule(c1=embedding_dim, c2=embedding_dim, k=3, p=1, act='relu')
        ])

        # PSP 部分：金字塔池化模块
        self.psp_modules = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=scale),
                ConvModule(c1=embedding_dim, c2=embedding_dim // 4, k=1, act='relu')
            ) for scale in pool_scales
        ])
        self.psp_fuse = ConvModule(
            c1=embedding_dim + (embedding_dim // 4) * len(pool_scales),
            c2=embedding_dim,
            k=3,
            p=1,
            act='relu'
        )

        # 最终融合和预测
        self.linear_fuse = ConvModule(c1=embedding_dim * 4, c2=embedding_dim, k=1, act='relu')
        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_ratio)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        n, _, h, w = c1.shape

        # MLP 投影并上采样到 C1 分辨率（与原 SegFormerHead 一致）
        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        # FPN 特征处理
        fpn_features = [_c1, _c2, _c3, _c4]
        fpn_outputs = [conv(feat) for conv, feat in zip(self.fpn_convs, fpn_features)]

        # PSP 模块：以 C4 的融合特征作为输入
        psp_input = fpn_outputs[-1]  # 用最深层特征
        psp_outs = [F.interpolate(psp(psp_input), size=psp_input.size()[2:], mode='bilinear', align_corners=False) 
                    for psp in self.psp_modules]
        psp_out = self.psp_fuse(torch.cat([psp_input] + psp_outs, dim=1))
        psp_out = F.interpolate(psp_out, size=c1.size()[2:], mode='bilinear', align_corners=False)

        # 融合所有 FPN 输出
        fused = self.linear_fuse(torch.cat(fpn_outputs, dim=1))

        # 结合 FPN 和 PSP 结果
        x = self.dropout(fused + psp_out)  # 残差连接
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




class SegFormerHead_aspp(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, num_classes=20, in_channels=[32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1):
        super(SegFormerHead_aspp, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        # Replace linear_fuse with ASPP
        self.aspp = ASPP(in_channels=embedding_dim * 4, 
                        out_channels=embedding_dim, mid_channels=256, 
                        atrous_rates=(6, 12, 18))

        # self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.downsample = HWD(in_ch=embedding_dim, out_ch=embedding_dim)
        self.c2psa = C2PSA(c1=embedding_dim, c2=embedding_dim, n=1, e=0.5)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_ratio)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        n = c4.shape[0]  # 批次大小

        # 处理每个特征层，确保通道数为embedding_dim
        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, self.embedding_dim, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, self.embedding_dim, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, self.embedding_dim, c2.shape[2], c2.shape[3])  # 修正
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, self.embedding_dim, c1.shape[2], c1.shape[3])

        # 拼接，通道数应为embedding_dim * 4
        cat_features = torch.cat([_c4, _c3, _c2, _c1], dim=1)
        
        # 调试：打印拼接后的形状
        # print("Concatenated feature map shape:", cat_features.shape)

        _c = self.aspp(cat_features)
        # x = self.downsample(_c)
        # x = self.c2psa(x)
        # x = self.upsample(x)

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x







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
        # self.decode_head = SegFormerHead_aspp(num_classes, self.in_channels, self.embedding_dim)
        # self.decode_head = SegFormerHead(num_classes, self.in_channels, self.embedding_dim)
        self.decode_head = SegFormerHead_aspp(num_classes, self.in_channels, self.embedding_dim)
        # self.decode_head = UPerHead(num_classes, self.in_channels, self.embedding_dim)
        self.c2psa_layers = nn.ModuleList([
            None,  # c1
            C2PSA(self.in_channels[1], self.in_channels[1], n=3, e=0.5),  # c2 使用 C2PSA
            None,  # c3
            C2PSA(self.in_channels[3], self.in_channels[3], n=3, e=0.5)   # c4 使用 C2PSA
        ])

        self.enhance_layers = nn.ModuleList([
            Bottleneck(self.in_channels[0], self.in_channels[0], shortcut=True, e=0.5),  # c1
            C2PSA(self.in_channels[1], self.in_channels[1], n=3, e=0.5),  # c2 使用 C2PSA
            # SCSABlock(self.in_channels[1], head_num= 8 ),  # c3
            SCSABlock(self.in_channels[2], head_num= 8 ),  # c3
            # SCSABlock(self.in_channels[3], head_num= 8 ),  # c3
            C2PSA(self.in_channels[3], self.in_channels[3], n=3, e=0.5)   # c4 使用 C2PSA
        ])

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        
        features = self.backbone.forward(inputs)
        # print(f"Backbone output shapes: {[f.shape for f in features]}")
        # x = self.decode_head.forward(x)
        # 对每个特征图应用 C2PSA
        # print(f"Enhanced features shapes: {[f.shape for f in enhanced_features]}")

        # enhanced_features = []
        # for i, (c2psa, feat) in enumerate(zip(self.c2psa_layers, features)):
        #     if c2psa is not None:  # 如果有 C2PSA 模块，则应用
        #         enhanced_features.append(c2psa(feat))
        #     else:  # 否则直接使用原始特征图
        #         enhanced_features.append(feat)
        # enhanced_features = [layer(feat) if layer else feat 
        #             for layer, feat in zip(self.enhance_layers, features)]

        # 传递给解码头
        x = self.decode_head(features)
        
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x
    




