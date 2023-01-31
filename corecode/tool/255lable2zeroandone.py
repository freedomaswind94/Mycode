# -*- coding: utf-8 -*-
"""
@File ：255lable2zeroandone.py
@Time ： 2023/1/16 0016 17:00
@Auth ： GiserLee
@E-mail：554758017@qq.com
@IDE ：PyCharm
"""
import glob
import numpy as np
from PIL import Image
import os
label_path = r'F:\dataset\LEVIRCD\CDLabDataset\val\OUT'
save_path = r'F:\dataset\LEVIRCD\CDLabDataset\val\label'

img_list = glob.glob(os.path.join(label_path,'*.png'))

for item in img_list:
    path,name = os.path.split(item)
    np_img = np.array(Image.open(item))
    two_img = Image.fromarray(np.uint8(np_img//255))
    two_img.save(os.path.join(save_path,name))
    print('finish')