# -*- coding: utf-8 -*-
"""
@File ：levircdTotxt.py
@Time ： 2023/1/10 0010 15:07
@Auth ： GiserLee
@E-mail：554758017@qq.com
@IDE ：PyCharm
"""
# Copyright (c) Open-CD. All rights reserved.
import argparse
import os
import os.path as osp
import glob

#匹配levircd数据集模式
def generate_txt_from_dir(src_dir, dst_dir, split,mode='train'):
    """Generate .txt file for LEVIR-CD dataset.

    Args:
        src_dir (str): path of the source dataset.
        dst_dir (str): Path to save .txt file.
        split (str): sub_dirs. 'train', 'val' or 'test'

    """
    if mode == 'train':
        src_dir = osp.join(src_dir, split)
        sub_dir_1 = osp.join(src_dir, 'A')
        sub_dir_2 = osp.join(src_dir, 'B')
        ann_dir = osp.join(src_dir, 'label')

        pre_a_imglist = sorted(glob.glob(osp.join(sub_dir_1,'*.png')))
        last_b_imglist = sorted(glob.glob(osp.join(sub_dir_2,'*.png')))
        ann_imglist = sorted(glob.glob(osp.join(ann_dir,'*.png')))

        assert len(pre_a_imglist) == len(last_b_imglist) == len(ann_imglist), '前后时相影像数量不匹配'

        with open('{}.txt'.format(osp.join(dst_dir, split)), 'w') as f:
            for i in range(len(pre_a_imglist)):
                temp = pre_a_imglist[i]+','+last_b_imglist[i]+','+ann_imglist[i]
                f.write(temp+'\n')

    else:
        src_dir = osp.join(src_dir, split)
        sub_dir_1 = osp.join(src_dir, 'A')
        sub_dir_2 = osp.join(src_dir, 'B')
        #ann_dir = osp.join(src_dir, 'label')

        pre_a_imglist = sorted(glob.glob(osp.join(sub_dir_1, '*.png')))
        last_b_imglist = sorted(glob.glob(osp.join(sub_dir_2, '*.png')))
        #ann_imglist = sorted(glob.glob(osp.join(ann_dir, '*.png')))

        assert len(pre_a_imglist) == len(last_b_imglist) , '前后时相影像数量不匹配'

        with open('{}.txt'.format(osp.join(dst_dir, split)), 'w') as f:
            for i in range(len(pre_a_imglist)):
                temp = pre_a_imglist[i] + ',' + last_b_imglist[i]
                f.write(temp+'\n')

def main():
    dataset_path = r'F:\dataset\LEVIRCD\CDLabDataset'
    out_dir = r'F:\dataset\LEVIRCD\CDLabDataset'
    print('Making .txt files ...')
    generate_txt_from_dir(dataset_path, out_dir, 'train')
    generate_txt_from_dir(dataset_path, out_dir, 'val')
    generate_txt_from_dir(dataset_path, out_dir, 'test',mode='test')
    print('Done!')

if __name__ == '__main__':
    main()
