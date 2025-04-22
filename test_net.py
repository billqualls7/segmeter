import os
import torch
import torch.nn as nn
import torch.optim as optim
from nets.segnext.model import SegNext
from nets.wyunet.unet import UNetV3_2
from nets.wyunet.dataload import Unet_SegmentationDataset

from tqdm import tqdm

from utils.dataloader import SegmentationDataset, seg_dataset_collate, DeeplabDataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from nets.segformer import SegFormer


def calculate_miou(confusion_matrix):
    """
    计算 mIoU（忽略背景类，假设背景类是0）
    Args:
        confusion_matrix: [num_classes, num_classes] 的混淆矩阵
    Returns:
        miou: 平均交并比（忽略背景类）
        iou_per_class: 每个类别的 IoU
    """
    # 计算交集（对角线）和并集（行和 + 列和 - 对角线）
    intersection = np.diag(confusion_matrix)
    union = confusion_matrix.sum(axis=1) + confusion_matrix.sum(axis=0) - intersection
    
    # 计算每个类别的 IoU，避免除以 0
    iou_per_class = np.where(union > 0, intersection / union, 0)
    
    # 忽略背景类（索引0）
    valid_classes = iou_per_class[1:]  # 排除背景类
    
    # 计算 mIoU（仅对有效类别取平均）
    miou = np.nanmean(valid_classes) if len(valid_classes) > 0 else 0
    
    return miou, iou_per_class 




num_classes = 3

input_shape = [640,640]
batch_size = 4
lr = 1e-4  # Learning rate
epochs = 10  # Number of epochs to train
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
background_class = 0 




VOCdevkit_path  = '../meter_seg_400'
with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"),"r") as f:
        train_lines = f.readlines()
with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),"r") as f:
    val_lines = f.readlines()

train_dataset   = Unet_SegmentationDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
val_dataset     = Unet_SegmentationDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=seg_dataset_collate)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=seg_dataset_collate)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# 检查第一批数据
for batch in train_loader:
    # images, pngs  = batch
    images, pngs, seg_labels = batch
    print(f"Images: {images.shape}, {images.dtype}, min: {images.min()}, max: {images.max()}")
    print(f"PNGs: {pngs.shape}, {pngs.dtype}, min: {pngs.min()}, max: {pngs.max()}")
    print(f"Seg_labels: {seg_labels.shape}, {seg_labels.dtype}, min: {seg_labels.min()}, max: {seg_labels.max()}")
    print("Pngs unique values:", np.unique(pngs.numpy()))  # 应为 [0, 1, 2]
    # print(seg_labels[0, :, 0, 0])
    break


# 定义网络
net = SegNext(num_classes=3, in_channnels=3, embed_dims=[32, 64, 6, 256],
              ffn_ratios=[4, 4, 4, 4], depths=[3, 3, 5, 2], num_stages=4,
              dec_outChannels=256, drop_path=0.2).cuda()

# phi             = "b0"
# net = SegFormer(num_classes=num_classes, phi=phi, pretrained=False).cuda()

# net = UNetV3_2(in_channels = 3, out_channels = num_classes).cuda()

# Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()  # For segmentation tasks
optimizer = optim.AdamW(net.parameters(), lr=lr)








for epoch in range(epochs):
    net.train()  # Set the model to training mode
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=100)

    for i, batch in enumerate(pbar):
        images, pngs, seg_labels = batch
        # images, pngs = batch
        images = images.to(device)
        # seg_labels = seg_labels.to(device)
        pngs = pngs.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = net(images)

        # Compute loss
        loss = criterion(outputs, pngs)
        loss.backward()
        optimizer.step()

        # Track the running loss
        running_loss += loss.item()

        # Update progress bar
        pbar.set_postfix(loss=f"{running_loss / (i+1):.4f}")

    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")

    # Validation step
    net.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for val_images, val_png, _ in val_loader:
            val_images = val_images.cuda()
            val_png = val_png.cuda()
            # print(val_png.shape)
            val_outputs = net(val_images)
            val_loss += criterion(val_outputs, val_png).item()
            
            # 获取预测类别（[B, H, W]）
            max_values, predicted = torch.max(val_outputs, 1)
            # 将预测结果转换为图像并保存
            batch_size = predicted.size(0)
            for j in range(batch_size):
                # 将tensor转换为numpy数组
                pred_array = predicted[j].cpu().numpy()
                
                # 如果你的预测是分割mask，通常需要转换为uint8格式
                # pred_array = pred_array.astype(np.uint8)
                pred_image = np.where(pred_array > 0, 255, 0).astype(np.uint8)
                
                # 如果你的类别需要映射到特定颜色，可以添加颜色映射
                # 例如对于二分类：
                # pred_image = pred_array * 255  # 二值化示例
                
                # 转换为PIL图像
                pred_pil = Image.fromarray(pred_image)
                
                # 保存预测图像
                output_dir = "debug/predictions"  # 输出目录
                os.makedirs(output_dir, exist_ok=True)
                save_path = os.path.join(output_dir, f"pred_image_{i}_{j}.png")
                pred_pil.save(save_path)
            # 更新混淆矩阵
            # for lt, lp in zip(val_png.flatten(), predicted.flatten()):
            #     confusion_matrix[lt.cpu(), lp.cpu()] += 1

    # 计算 mIoU（忽略背景类）
    # miou, iou_per_class = calculate_miou(confusion_matrix)

    # print(f"Validation Loss: {val_loss / len(val_loader):.4f}")
    # print(f"Per-Class IoU: {iou_per_class}")
    # print(f"mIoU (Ignore Background): {miou:.4f}")
        # print(f"Validation Loss: {val_loss / len(val_loader):.4f}")
        # print(f"Validation Pixel Accuracy: {100 * correct_pixels / total_pixels:.2f}%")
        
    # Save model after every epoch
    # torch.save(net.state_dict(), f"segnext_epoch{epoch+1}.pth")

print("Training complete.")








# # 生成合成数据
# batch_size = 4
# h, w = 640, 640
# inputs = torch.randn(batch_size, 3, h, w).cuda()
# labels = torch.randint(0, 3, (batch_size, h, w)).cuda()

# # 定义损失和优化器
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.01)

# # 训练
# net.train()
# for epoch in range(50):
#     optimizer.zero_grad()
#     outputs = net(inputs)
#     loss = criterion(outputs, labels)
#     loss.backward()
#     optimizer.step()
#     print(f"Epoch {epoch+1}/500, Loss: {loss.item():.4f}")

# # 检查输出形状
# print(f"Output shape: {outputs.shape}")