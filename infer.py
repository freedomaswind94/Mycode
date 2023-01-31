# -*- coding: utf-8 -*-
"""
@File ：infer.py
@Time ： 2023/1/17 0017 18:07
@Auth ： GiserLee
@E-mail：554758017@qq.com
@IDE ：PyCharm
"""
import torch
import cv2
import numpy as np
import os.path as osp
import corecode.dataset.transforms as tr
from tqdm import tqdm
from PIL import Image
from corecode.models.UNet_CD import Unet
from torch.utils.data import Dataset, DataLoader
# 加载模型
def resume_model(model,inputimage_CH,classNum,model_save_path,device):
    model = model(input_nbr=inputimage_CH, label_nbr=classNum)
    model_dict = torch.load(model_save_path)
    model.load_state_dict(model_dict['model_state_dict'])
    model.to(device)
    return model

"""choince transform by yourself here"""
mean,std=(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
infer_transPipeline = [tr.toTensor(),tr.transforms.Normalize(mean=mean, std=std)]
#lable_transPipeline = [tr.toTensor()]
def transPipeline(*args):
    A, B = args
    img1 = tr.transComposePipeline(infer_transPipeline)(A)
    img2 = tr.transComposePipeline(infer_transPipeline)(B)
    return img1,img2
# 仅获取 A B影像
class CDdataSet(Dataset):
    def __init__(self, data_path):
        super(CDdataSet, self).__init__()
        self.data_path = data_path
        self.data_txt_path = osp.join(data_path,'test.txt')
        self.fpath = self.getDataPath(self.data_txt_path)
        self.transPipeline = transPipeline

    def __getitem__(self, index):
        image_A_paths, image_B_paths = self.fpath[index][0].strip('\n').split(',')
        A = self.imageOpenToNdarray(image_A_paths)
        B = self.imageOpenToNdarray(image_B_paths)
        imageA, imageB = transPipeline(A,B)
        return imageA, imageB

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
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    data_path = r'F:\dataset\LEVIRCD\CDLabDataset'
    model_save_path = r'F:\changedetection_code\Myconde\corecode\save\model\cdUnet.pth'
    result_save_path = r'corecode/save/picture'
    batch_size = 1
    input_channels = 3
    class_num = 2
    infer_dataset = CDdataSet(data_path=data_path)
    infer_dataLoader = DataLoader(dataset=infer_dataset,batch_size=batch_size,shuffle=False,num_workers=1)
    model = resume_model(model=Unet, inputimage_CH=input_channels * 2, classNum=class_num
                         , model_save_path=model_save_path, device=DEVICE)
    model.eval()
    with torch.no_grad():
        pbar = tqdm(infer_dataLoader)
        for i,(A,B) in enumerate(pbar):
            pbar.set_description('star val please waite')
            A, B = A.to(DEVICE, dtype=torch.float), B.to(DEVICE, dtype=torch.float)
            out = model(A,B)
            imgPredict = torch.argmax(out, dim=1)
            imgPredict = np.squeeze(imgPredict.cpu().numpy(),axis=0)
            # 保存为0 255像素结果
            cv2.imwrite(osp.join(result_save_path,'{}.png'.format(i)),imgPredict*255)
