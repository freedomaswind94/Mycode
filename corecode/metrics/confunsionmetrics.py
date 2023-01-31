# -*- coding: utf-8 -*-
"""
@File ：confunsionmetrics.py
@Time ： 2023/1/13 0013 10:34
@Auth ： GiserLee
@E-mail：554758017@qq.com
@IDE ：PyCharm
"""
import torch
from PIL import Image
from pathlib import Path
import numpy as np

# 测试通过
"""
refer to https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/metrics.py
"""

"""
confusionMetric
L\P     P    N

P      TP    FN

N      FP    TN

"""
class CDMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def overallAccuracy(self):
        # return all class overall pixel accuracy,AO评价指标
        # acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=0) + np.sum(self.confusionMatrix, axis=1) - np.diag(
            self.confusionMatrix)
        IoU = intersection / union
        mIoU = np.nanmean(IoU)
        return mIoU

    def precision(self):
        # precision = TP / TP + FP
        """return unchange and change precision list"""
        p = np.diag(self.confusionMatrix) / (self.confusionMatrix.sum(axis=0)+1e-4)
        return p[1]

    def recall(self):
        # recall = TP / TP + FN
        """return unchange and change recall list"""
        r = np.diag(self.confusionMatrix) / (self.confusionMatrix.sum(axis=1)+1e-4)
        return r[1]

    def genConfusionMatrix(self, imgPredict, imgLabel):
        # remove classes from unlabeled pixels in gt image and predict
        numClass = self.numClass
        assert imgPredict.shape == imgLabel.shape, 'error'

        def __fast_hist(label_gt, label_pred):
            mask = (label_gt >= 0) & (label_gt < numClass)
            hist = np.bincount(numClass * label_gt[mask].astype(int) + label_pred[mask],
                               minlength=numClass ** 2).reshape(numClass, numClass)
            return hist

        confusion_matrix = np.zeros((self.numClass, self.numClass))
        for lt, lp in zip(imgLabel, imgPredict):
            confusion_matrix += __fast_hist(lt.flatten(), lp.flatten())
        return confusion_matrix

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape

        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)
    def addBatchNew(self, imgPredict, imgLabel):

        imgPredict = torch.argmax(imgPredict, dim=1) # b  h w
        imgPredict = imgPredict.cpu().numpy()
        imgLabel = imgLabel.cpu().numpy() # b h w
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))

    def pre_reca_fscor(self):
        pre = self.precision()
        reca = self.recall()
        fscor = (2 * pre * reca) / (pre + reca+1e-4)
        return list([pre,reca,fscor])


if __name__ == '__main__':
    true_path = r'F:\dataset\LEVIRCD\me\a'
    pred_path = r'F:\dataset\LEVIRCD\me\b'
    class_num = 2
    metric = CDMetric(class_num)
    true_list = Path(true_path).rglob('*.png')
    pred_list = Path(pred_path).rglob('*.png')

    for true, pred in zip(true_list, pred_list):
        true_img = np.array(Image.open(true), dtype=np.uint8)
        #单波段标签需要编码成0,1，2.。。。
        true_img = true_img//255
        pred_img = np.array(Image.open(pred), dtype=np.uint8)
        pred_img = pred_img // 255
        metric.addBatch(pred_img, true_img)
        L = metric.pre_reca_fscor()
        print(L)
    oa = metric.overallAccuracy()
    mIoU = metric.meanIntersectionOverUnion()
    p = metric.precision()
    mp = np.nanmean(p)
    r = metric.recall()
    mr = np.nanmean(r)
    f1 = (2 * p * r) / (p + r)
    mf1 = np.nanmean(f1)

    print(f'类别0,类别1,...\n  oa:{oa}, mIou:{mIoU}, p:{p}, mp:{mp}, r:{r}, mr:{mr}, f1:{f1}, mf1:{mf1}')
