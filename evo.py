'''
Author: wuyao 1955416359@qq.com
Date: 2025-01-09 16:19:49
LastEditors: wuyao 1955416359@qq.com
LastEditTime: 2025-03-31 17:57:14
FilePath: /wygraduate/segtask/evo.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import colorsys
import copy
import time
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from tqdm import tqdm
from utils.utils_metrics import compute_mIoU, show_results


# from nets.deeplabv3_plus import DeepLab
# from nets.hrnet.hrnet import HRnet
# from nets.pspnet.pspnet import PSPNet
# from nets.unet.unet import Unet

from nets.segformer import SegFormer



from utils.utils import cvtColor, preprocess_input, resize_image, show_config
from Colormap import Colormap


class EVO():
    def __init__(self):
        #----------------------------------------#
        #   所需要区分的类的个数+1
        #----------------------------------------#
        self.num_classes = 3
        self.input_shape = [640,640]
        self.model_path =  'logs/deeplabv3_meter_seg_400/deeplabv3_meter_seg_400_best_epoch_weights.pth'
        self.downsample_factor = 16
        #-------------------------------------------------#
        #   mix_type参数用于控制检测结果的可视化方式
        #
        #   mix_type = 0的时候代表原图与生成的图进行混合
        #   mix_type = 1的时候代表仅保留生成的图
        #   mix_type = 2的时候代表仅扣去背景，仅保留原图中的目标
        #-------------------------------------------------#
        self.mix_type = 0
        self.device      =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print(self.device)

        if self.device:
            self.cuda = True
            print(self.device)
        else: self.cuda = False
        # self.cuda = False

        self.colormap = Colormap()

    def generate_network(self, net_type = 'deeplabv3'):
        self._init_net_(net_type =net_type )
        
        self.net.load_state_dict(torch.load(self.model_path, map_location=self.device))
        if self.device: self.net.to(self.device)
        self.net    = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))



    def _init_net_(self, net_type = 'SegFormer'):
        # if net_type =="deeplabv3":
        #     self.net_backbone = "mobilenet"
        #     # default  backbone        = "mobilenet"
        #     self.net = DeepLab(num_classes=self.num_classes, backbone=self.net_backbone, downsample_factor=self.downsample_factor, pretrained=False)
       
        # elif net_type == "hrnet":
        #     self.net_backbone = "hrnetv2_w18"
        #     # default  backbone        = "hrnetv2_w18"
        #     self.net = HRnet(num_classes=self.num_classes, backbone=self.net_backbone, pretrained=False)
        
        # elif net_type == "pspnet":
        #     # default  backbone        = "mobilenet"  resnet

        #     self.net_backbone = "mobilenet"
        #     self.net = PSPNet(num_classes=self.num_classes, backbone=self.net_backbone, downsample_factor=self.downsample_factor, pretrained=False, aux_branch=False)

        # elif net_type == "unet":
        #     # default  backbone        = "vgg"  resnet
        #     self.net_backbone = "vgg"
        #     self.net = Unet(num_classes=self.num_classes, backbone=self.net_backbone, pretrained=False)

        if net_type =="SegFormer":
            self.net_backbone = "SegFormer"
            # default  backbone        = "mobilenet"
            self.net = SegFormer(num_classes=num_classes, pretrained=False)


        else:
            assert False, "Wrong Network"
        

    def get_miou_png(self, image):
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data, nw, nh  = resize_image(image, (self.input_shape[1],self.input_shape[0]))
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
                
            #---------------------------------------------------#
            #   图片传入网络进行预测
            #---------------------------------------------------#
            pr = self.net(images)[0]
            #---------------------------------------------------#
            #   取出每一个像素点的种类
            #---------------------------------------------------#
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()
            #--------------------------------------#
            #   将灰条部分截取掉
            #--------------------------------------#
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
            #---------------------------------------------------#
            #   进行图片的resize
            #---------------------------------------------------#
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
            #---------------------------------------------------#
            #   取出每一个像素点的种类
            #---------------------------------------------------#
            pr = pr.argmax(axis=-1)
            # print(pr.shape)
            colored_pr = np.zeros((pr.shape[0], pr.shape[1], 3), dtype=np.uint8)

            for i in range(pr.shape[0]):
                for j in range(pr.shape[1]):
                    class_id = pr[i, j]
                    colored_pr[i, j] = self.colormap.get_color(class_id)

            # 转换为PIL图像返回
        image = Image.fromarray(colored_pr)
        grayimage = Image.fromarray(np.uint8(pr))
        return image ,grayimage





if __name__ == "__main__":
    #---------------------------------------------------------------------------#
    #   miou_mode用于指定该文件运行时计算的内容
    #   miou_mode为0代表整个miou计算流程，包括获得预测结果、计算miou。
    #   miou_mode为1代表仅仅获得预测结果。
    #   miou_mode为2代表仅仅计算miou。
    #---------------------------------------------------------------------------#
    miou_mode       = 0
    #------------------------------#
    #   分类个数+1、如2+1
    #------------------------------#
    num_classes     = 3
    name_classes    = ["_background_","pointer","dial"]


    net_type = 'SegFormer'
    model_path = "model_data/segformer_c2psa_best_epoch_weights.pth"


    VOCdevkit_path  = '../meter_seg_400'
    image_ids       = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),'r').read().splitlines() 
    gt_dir          = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/")
    miou_out_path   = "miou_out/" + net_type
    gray_pred_dir        = os.path.join(miou_out_path, 'grayresults')
    pred_dir = os.path.join(miou_out_path, 'rgbresults')

    if miou_mode == 0 or miou_mode == 1:
        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(gray_pred_dir, exist_ok=True)


    
    # model_path = "logs/deeplabv3_meter_seg_400/deeplabv3_meter_seg_400_best_epoch_weights.pth"


    save = True
    

    evo =  EVO()
    evo.model_path = model_path
    print("Load model.")
    evo.generate_network(net_type)
    print("Load model done.")

    print("Get predict result.")


    for image_id in tqdm(image_ids):
            image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
            image       = Image.open(image_path)
            image, grayimage       = evo.get_miou_png(image)
            if save :
                image.save(os.path.join(pred_dir, image_id + ".png"))
                grayimage.save(os.path.join(gray_pred_dir, image_id + ".png"))
    print("Get predict result done.")
    

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, gray_pred_dir, image_ids, num_classes, name_classes)  # 执行计算mIoU的函数
   
        print("Get miou done.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)
