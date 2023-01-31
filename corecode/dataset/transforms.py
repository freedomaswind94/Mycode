# -*- coding: utf-8 -*-
"""
@File ：transforms.py
@Time ： 2023/1/10 0010 16:00
@Auth ： GiserLee
@E-mail：554758017@qq.com
@IDE ：PyCharm
"""
import torch
import torchvision.transforms as transforms
import albumentations as A
"""
define customers transformer
"""

def albtrans(img_A,img_B,label):
    '''
    use albumentation augment Data
    used in train mode
    :param img_A: pre img
    :param img_B: last img
    :param label: change label
    :return: transed data
    '''
    trans = A.Compose([
        #输入网络的图片尺寸，要根据网络修改
        A.PadIfNeeded(p=1, min_height=256, min_width=256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=60,p=0.3),
        A.OneOf([
            # A.IAAAdditiveGaussianNoise(),   # 将高斯噪声添加到输入图像
            A.GaussNoise(),  # 将高斯噪声应用于输入图像。
        ], p=0.3),  # 应用选定变换的概率
        A.OneOf([
            A.MotionBlur(p=0.2),  # 使用随机大小的内核将运动模糊应用于输入图像。
            A.MedianBlur(blur_limit=3, p=0.1),  # 中值滤波
            A.Blur(blur_limit=3, p=0.1),  # 使用随机大小的内核模糊输入图像。
        ], p=0.3),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.3),
        # 随机应用仿射变换：平移，缩放和旋转输入
        A.RandomBrightnessContrast(p=0.2),  # 随机明亮对比度
    ], additional_targets={'image1': 'image'})
    transData = trans(image=img_A, image1=img_B, mask=label)
    return transData['image'],transData['image1'],transData['mask']


def transComposePipeline(tranlist):
    com = transforms.Compose(
        tranlist
    )
    return com

class mask2onehot(object):
    def __call__(self, img):
        """
           img: 标签图像 # （ 1, h, w)  ==> (N h w) N = num_class
        """
        num_classes = 2  # 分类类别数

        current_label = img.squeeze()  # （1, h, w) ---> （h, w)

        h, w = current_label.shape[0], current_label.shape[1]

        tmplate = torch.ones(num_classes, h, w)
        for i in range(num_classes):
            tmplate[i][current_label != i] = 0

        return tmplate


# to 直接实现 hwc 转 chw
class toTensor(object):
    def __call__(self, img):
        return transforms.ToTensor()(img)


class norMalize(object):
    def __call__(self, img):
        return transforms.Normalize()(img)


class dataTransPose(object):
    def __call__(self, img):
        return img.transpose(0, 3)


class npTotensor(object):
    def __call__(self, img):
        return torch.from_numpy(img)


if __name__ == '__main__':
    from PIL import Image
    import numpy as np

    datapath = r'F:\dataset\LEVIRCD\test\A\test_1.png'
    img = np.array(Image.open(datapath))
    print(img.shape)
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    transComposePipeline = transComposePipeline([toTensor(), transforms.Normalize(mean=mean, std=std)])
    ts = transComposePipeline(img)
    print(ts.shape)

    # n = np.random.randn(8,5,4,3)
    # m=toTensor()(n)
    # print(m.shape)
