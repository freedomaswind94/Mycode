# -*- coding: utf-8 -*-
"""
@File ：CDdataSet.py
@Time ： 2023/1/10 0010 14:50
@Auth ： GiserLee
@E-mail：554758017@qq.com
@IDE ：PyCharm
"""
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import corecode.dataset.transforms as tr

"""choince transform by yourself here"""
mean,std=(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
train_transPipeline = [tr.toTensor(),tr.transforms.Normalize(mean=mean, std=std)]
vel_transPipeline = [tr.toTensor(),tr.transforms.Normalize(mean=mean, std=std)]
lable_transPipeline = [tr.toTensor()]


def transPipeline(test_mode=False,*args):
    A, B, lable = args
    """mode train"""
    if not test_mode:
        img_A, img_B, mask = tr.albtrans(A,B,lable)
        img1 = tr.transComposePipeline(train_transPipeline)(img_A)
        img2 = tr.transComposePipeline(train_transPipeline)(img_B)
        lable = tr.transComposePipeline(lable_transPipeline)(mask)
        return img1,img2,lable
    """mode val"""
    if test_mode:
        img1 = tr.transComposePipeline(vel_transPipeline)(A)
        img2 = tr.transComposePipeline(vel_transPipeline)(B)
        lable = tr.transComposePipeline(lable_transPipeline)(lable)
        return img1,img2,lable

#针对levir-cd数据集
class CDdataSet(Dataset):
    """
    data_path:(str)  path of data
    modeOfdir:(str) one of in [train , val]
    test_mode:(bool)
    """
    def __init__(self, data_path, modeOfdir,test_mode=False):
        super(CDdataSet, self).__init__()
        self.data_path = data_path
        self.test_mode = test_mode
        self.data_txt_path = data_path + '\\'+ modeOfdir + '.txt'
        self.fpath = self.getDataPath(self.data_txt_path)
        self.transPipeline = transPipeline


    def __getitem__(self, index):

        """mode is train """
        if not self.test_mode:
            image_A_paths, image_B_paths, label_paths = self.fpath[index][0].strip('\n').split(',')

            # H W C
            A = self.imageOpenToNdarray(image_A_paths)
            B = self.imageOpenToNdarray(image_B_paths)
            label = self.imageOpenToNdarray(label_paths)
            """
            transform
            """
            # C H W
            imageA, imageB, label = transPipeline(self.test_mode,A,B,label)
            return imageA, imageB, label
        """mode is val"""
        if self.test_mode:
            image_A_paths, image_B_paths,label_paths = self.fpath[index][0].strip('\n').split(',')

            A = self.imageOpenToNdarray(image_A_paths)
            B = self.imageOpenToNdarray(image_B_paths)
            label = self.imageOpenToNdarray(label_paths)
            imageA, imageB, label = transPipeline(self.test_mode,A,B,label)
            return imageA, imageB, label

    def getDataPath(self,data_txt_path):
        fpath=[]
        f=open(data_txt_path)
        for line in f:
            fpath.append([line])
        f.close()
        return fpath

    def imageOpenToNdarray(self,img_path_list):
        img = np.array(Image.open(img_path_list))
        return img

    def __len__(self):
        return len(self.fpath)

if __name__ == '__main__':
    CCD = CDdataSet(r'F:\dataset\LEVIRCD\CDLabDataset','train',test_mode=True)
    a=CCD[1]
    print('success')