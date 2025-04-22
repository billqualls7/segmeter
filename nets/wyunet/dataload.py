import os
import numpy as np
from PIL import Image
import cv2
from torch.utils.data import Dataset
import torch

class Unet_SegmentationDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        """
        初始化数据集
        :param annotation_lines: 包含图像文件名的列表（每行一个文件名）
        :param input_shape: 目标尺寸 [H, W]
        :param num_classes: 类别数（包括背景，例如 3）
        :param train: 是否为训练集（True 表示训练，False 表示验证）
        :param dataset_path: 数据集根目录（如 VOCdevkit）
        """
        super(Unet_SegmentationDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape  # e.g., [256, 256]
        self.num_classes = num_classes  # e.g., 3
        self.train = train
        self.dataset_path = dataset_path

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # 获取图像和标签路径
        annotation_line = self.annotation_lines[index]
        name = annotation_line.split()[0]
        jpg_path = os.path.join(self.dataset_path, "VOC2007/JPEGImages", name + ".jpg")
        png_path = os.path.join(self.dataset_path, "VOC2007/SegmentationClass", name + ".png")

        # 读取图像和标签
        jpg = Image.open(jpg_path)
        png = Image.open(png_path)

        # 数据增强和预处理
        jpg, png = self.get_random_data(jpg, png, self.input_shape, random=self.train)

        # 转换为 numpy 数组
        jpg = np.array(jpg, dtype=np.float32)  # [H, W, 3]
        png = np.array(png, dtype=np.uint8)    # [H, W]

        # 处理标签中的无效值
        png[png >= self.num_classes] = 0  # 将超出类别范围的值设为背景（0）

        # 图像预处理
        jpg = jpg / 255.0  # 归一化到 [0, 1]
        jpg = np.transpose(jpg, (2, 0, 1))  # 调整为 [3, H, W]

        # 生成 one-hot 编码标签
        labels = np.eye(self.num_classes + 1)[png]  # [H, W, num_classes]
        labels = np.transpose(labels, (2, 0, 1))  # [num_classes, H, W]
        labels  = labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))
        # 转换为 torch 张量
        imgs = torch.tensor(jpg, dtype=torch.float32)      # [3, H, W]
        pngs = torch.tensor(png, dtype=torch.long)         # [H, W]
        labels = torch.tensor(labels, dtype=torch.float32) # [num_classes, H, W]
        # print(f"Imgs shape: {imgs.shape}, Pngs shape: {pngs.shape}, Labels shape: {labels.shape}")
        # print(f"Labels unique: {torch.unique(labels)}")  # 应为 [0, 1]
        return imgs, pngs, labels

    def rand(self, a=0, b=1):
        """生成随机数"""
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, random=True):
        """
        数据增强和预处理
        :param image: PIL RGB 图像
        :param label: PIL 标签图像
        :param input_shape: 目标尺寸 [H, W]
        :param random: 是否应用随机增强
        """
        image = self.cvtColor(image)  # 确保 RGB 格式
        label = Image.fromarray(np.array(label))  # 确保标签是 PIL 图像
        iw, ih = image.size
        h, w = input_shape

        if not random:
            # 验证模式：仅缩放和填充
            scale = min(w / iw, h / ih)
            nw, nh = int(iw * scale), int(ih * scale)
            image = image.resize((nw, nh), Image.BICUBIC)
            label = label.resize((nw, nh), Image.NEAREST)

            new_image = Image.new('RGB', (w, h), (128, 128, 128))  # 灰色填充
            new_label = Image.new('L', (w, h), 0)  # 背景填充为 0
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
            new_label.paste(label, ((w - nw) // 2, (h - nh) // 2))
            return new_image, new_label

        # 训练模式：随机增强
        # 随机缩放
        new_ar = iw / ih * self.rand(0.7, 1.3) / self.rand(0.7, 1.3)
        scale = self.rand(0.5, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)
        label = label.resize((nw, nh), Image.NEAREST)

        # 随机翻转
        if self.rand() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        # 填充
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_label = Image.new('L', (w, h), 0)
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))

        return new_image, new_label

    def cvtColor(self, image):
        """确保图像是 RGB 格式"""
        if image.mode != 'RGB':
            return image.convert('RGB')
        return image