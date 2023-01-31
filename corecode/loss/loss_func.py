# -*- coding: utf-8 -*-
"""
@File ：loss_func.py
@Time ： 2023/1/30 0030 11:16
@Auth ： GiserLee
@E-mail：554758017@qq.com
@IDE ：PyCharm
"""
from .dice_loss import dice_loss,dicloss_1
from .focal_loss import FocalLoss
import torch.nn.functional as F

class HybirdLoss():
    def __init__(self,class_num,is_FCDic=True):
        super(HybirdLoss, self).__init__()
        self.class_num = class_num
        self.is_FCDic = is_FCDic
    def __call__(self, predictions, target):
        is_FCDic = self.is_FCDic
        if is_FCDic:
            return self.hybrid_FC_Dice_loss(predictions=predictions, target=target,class_num=self.class_num)
        else:
            return self.hybrid_BCE_Dice_loss(predictions=predictions, target=target,class_num=self.class_num)

    def hybrid_BCE_Dice_loss(self,predictions, target,class_num=2):
        """Calculating the loss"""
        # gamma=0, alpha=None --> CE
        focal = FocalLoss(gamma=0, alpha=None, size_average=True)(predictions, target)
        dice = dice_loss(predictions,F.one_hot(target, class_num).permute(0, 3, 1, 2).float(), multiclass=False)
        loss = focal + dice
        return loss

    def hybrid_FC_Dice_loss(self,predictions, target,class_num=2):
        """Calculating the loss"""
        # gamma=0, alpha=None --> CE
        focal = FocalLoss(gamma=2, alpha=0.25, size_average=True)(predictions, target)
        dice = dice_loss(predictions,F.one_hot(target, class_num).permute(0, 3, 1, 2).float(), multiclass=False)
        # dice = dicloss1(predictions,F.one_hot(target, class_num).permute(0, 3, 1, 2).float())

        loss = focal + dice

        return loss

if __name__ == '__main__':
    pass