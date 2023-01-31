# -*- coding: utf-8 -*-
"""
@File ：seed.py
@Time ： 2023/1/12 0012 15:00
@Auth ： GiserLee
@E-mail：554758017@qq.com
@IDE ：PyCharm
"""
import torch
import random
import numpy as np
# 固定随机种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True