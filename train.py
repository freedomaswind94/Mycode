# -*- coding: utf-8 -*-
"""
@File ：train.py
@Time ： 2023/1/10 0010 14:41
@Auth ： GiserLee
@E-mail：554758017@qq.com
@IDE ：PyCharm
"""
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from corecode.dataset.CDdataSet import CDdataSet
from corecode.dataset.CDdataLoader import CDDataLoader
from corecode.models import UNet_CD,UNet_ASPP,SiamUnet_conc
from corecode.loss.loss_func import HybirdLoss
from corecode.metrics.confunsionmetrics import CDMetric
from torch.optim import lr_scheduler


# 加载模型,确定是否使用初始化方法
def load_model(model,inputimage_CH,classNum,device,is_init=True):
    '''
    inputimage_CH:
        if is siam type model,inputimage_CH=input_channels (3)
        else inputimage_CH=input_channels*2 (6)
    '''
    model = model(input_nbr=inputimage_CH, label_nbr=classNum)
    if is_init:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    model.to(device)
    return model

#保存模型
def save_model(exp_dir, model, optimizer):
    """
    Model Saving Function
    Args:
        exp_dir  (Path)   : model saving path
        model     : torch model
        optimizer : torch optimizer
    """
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimize_state_dict': optimizer.state_dict(),
        },
        exp_dir
    )


def single_train(model,data_loader,optimizer,loss_fun,device):
    """
    Train Function for a single Epoch
    """
    model.train()
    len_loader = len(data_loader)
    total_loss = 0.
    criterion_list = []
    train_metrics = CDMetric(numClass=2)
    #for i,(A, B, Target) in enumerate(data_loader):
    pbar = tqdm(data_loader)
    for i,(A, B, Target) in enumerate(pbar):
        pbar.set_description(f'Train Iter good luck:')
        A, B, Target = A.to(device,dtype=torch.float), B.to(device,dtype=torch.float), Target.to(device,dtype=torch.long)
        optimizer.zero_grad()
        output = model(A,B)
        Target = torch.squeeze(Target,dim=1) # b 1 h w => b h w
        loss = loss_fun(output, Target)
        #loss = FocalLoss()(output, Target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        """精度评定"""
        train_metrics.reset()
        train_metrics.addBatchNew(imgPredict=output,imgLabel=Target)
        criterion_list.append(train_metrics.pre_reca_fscor())
        if i % 2 == 0:
            iter_precision,iter_recall,iter_f1 =np.sum(criterion_list,axis=0)
            pbar.set_postfix({'Loss': total_loss / (i+1),'Pre': iter_precision / (i+1)
                                 ,'Recall': iter_recall / (i+1),'F1': iter_f1 / (i+1)})
    total_loss /= len_loader
    # epoc_precision,epoc_recall,epoc_f1=np.sum(criterion_list,axis=0)
    # print(f'train result-Iter :{len_loader}/{len_loader} Loss :{total_loss:.3f} Pre :{epoc_precision/len_loader:.3f} F1 :{epoc_f1/len_loader:.3f} R :{epoc_recall/len_loader:.3f}')

    return total_loss

def single_val(model,data_loader,loss_fun,device):
    """
    Validation Function for a single Epoch
    """
    model.eval()
    len_loader = len(data_loader)
    total_loss = 0.
    criterion_list = []
    val_metrics = CDMetric(numClass=2)
    with torch.no_grad():
        pbar = tqdm(data_loader)
        for i, (A, B, Target) in enumerate(pbar):
            pbar.set_description('star val please waite')
            A, B, Target = A.to(device,dtype=torch.float), B.to(device,dtype=torch.float), Target.to(device,dtype=torch.long)
            output = model(A, B)
            Target = torch.squeeze(Target, dim=1)
            loss = loss_fun(output, Target)
            total_loss += loss.item()

            val_metrics.reset()
            val_metrics.addBatchNew(imgPredict=output, imgLabel=Target)
            criterion_list.append(val_metrics.pre_reca_fscor())
            if (i+1) == len_loader:
                epoc_precision, epoc_recall, epoc_f1 = np.sum(criterion_list, axis=0)
                pbar.set_postfix({'Loss':total_loss/len_loader,'Pre':epoc_precision / len_loader,'F1':epoc_f1 / len_loader,'Rrcall':epoc_recall / len_loader})
                #print(f'val result-Iter :{len_loader}/{len_loader} Loss :{total_loss:.3f} Pre :{epoc_precision / len_loader:.3f} F1 :{epoc_f1 / len_loader:.3f}  Rrcall :{epoc_recall / len_loader:.3f}')
        total_loss /= len_loader
        lossval = [total_loss,epoc_precision/len_loader,epoc_f1/len_loader,epoc_recall/len_loader]

        return lossval

if __name__ == '__main__':
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    data_path = r'F:\dataset\LEVIRCD\CDLabDataset'
    model_save_path = r'F:\changedetection_code\Myconde\corecode\save\model\cdUnet.pth'
    batch_size = 2
    input_channels = 3
    class_num = 2
    lr_rate = 5e-3
    num_epochs = 100
    train_dataset = CDdataSet(data_path=data_path,modeOfdir='train',test_mode=False)
    val_dataset = CDdataSet(data_path=data_path,modeOfdir='val',test_mode=True)
    train_dataLoader = CDDataLoader(dataset=train_dataset,batch_size=batch_size)
    val_dataLoader = CDDataLoader(dataset=val_dataset,batch_size=batch_size)

    model = load_model(model=UNet_CD,inputimage_CH=input_channels*2,classNum=class_num,device=DEVICE)
    # criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
    criterion = HybirdLoss(class_num=class_num)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    best_loss_precision = 0
    for i in range(num_epochs):
        single_train(model=model,data_loader=train_dataLoader,optimizer=optimizer
                     ,loss_fun=criterion,device=DEVICE)

        scheduler.step()
        valloss = single_val(model=model,data_loader=val_dataLoader,loss_fun=criterion,device=DEVICE)
        precision = valloss[1]
        """model save """
        if best_loss_precision < precision:
            best_loss_precision = precision
            save_model(exp_dir=model_save_path,model=model,optimizer=optimizer)
            print("valid precision is improved. the model is saved.")
