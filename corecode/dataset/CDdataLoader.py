# -*- coding: utf-8 -*-
"""
@File ：CDdataLoader.py
@Time ： 2023/1/10 0010 14:51
@Auth ： GiserLee
@E-mail：554758017@qq.com
@IDE ：PyCharm
"""
from torch.utils.data import DataLoader
from corecode.dataset.CDdataSet import CDdataSet


def CDDataLoader(dataset,batch_size):
    """
    waiting impooving,now i m tinnking.....
    """
    Data_Loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,
                            num_workers=0)

    return Data_Loader

if __name__ == '__main__':
    CCD = CDdataSet(r'F:\dataset\LEVIRCD','val',test_mode=True)
    BATCHSIZE = 4
    valdata = CDDataLoader(CCD,BATCHSIZE)
    print(len(valdata))
    count = 0

    for i,X in enumerate(valdata):
        print(i)
        A,B,L=X
        print(
            A.shape,
            B.shape,
            L.shape,
        )
        count +=1
    print('total num is {}'.format(count*4))