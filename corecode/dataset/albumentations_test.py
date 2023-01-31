# -*- coding: utf-8 -*-
"""
@File ：albumentations_test.py
@Time ： 2023/1/28 0028 9:43
@Auth ： GiserLee
@E-mail：554758017@qq.com
@IDE ：PyCharm
"""

'''测试通过，待更新'''

import albumentations as A
import cv2
import numpy as np
from PIL import Image
img1_path = r'F:\dataset\LEVIRCD\CDLabDataset\processdata\train\A\train_2_17.png'
img2_path = r'F:\dataset\LEVIRCD\CDLabDataset\processdata\train\B\train_2_17.png'
mask_path = r'F:\dataset\LEVIRCD\CDLabDataset\processdata\train\OUT\train_2_17.png'

img1  =np.array(cv2.imread(img1_path))
img2  =np.array(cv2.imread(img2_path))
mask = np.array(Image.open(mask_path))

trans = A.Compose([
        # A.HorizontalFlip(p=1),
        # A.VerticalFlip(p=1),
        A.RandomRotate90(p=1),
        A.CenterCrop(p=1.0, height=100,
                    width=100),
        A.OneOf([
            # A.IAAAdditiveGaussianNoise(),   # 将高斯噪声添加到输入图像
            A.GaussNoise(),    # 将高斯噪声应用于输入图像。
        ], p=1),   # 应用选定变换的概率
        A.OneOf([
            A.MotionBlur(p=0.2),   # 使用随机大小的内核将运动模糊应用于输入图像。
            A.MedianBlur(blur_limit=3, p=0.1),    # 中值滤波
            A.Blur(blur_limit=3, p=0.1),   # 使用随机大小的内核模糊输入图像。
        ], p=1),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        # 随机应用仿射变换：平移，缩放和旋转输入
        A.RandomBrightnessContrast(p=0.2),   # 随机明亮对比度
    ],additional_targets={'image1': 'image'})

transData = trans(image=img1,image1=img2,mask=mask)
def visualize(image):
    # im=cv2.resize(image, (256, 256))
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image',image)
    cv2.waitKey()
visualize(transData['image'])
visualize(transData['image1'])
visualize(transData['mask'])
