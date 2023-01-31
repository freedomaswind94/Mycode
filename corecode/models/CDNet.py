import torch
import torch.nn as nn
import torch.nn.functional as F

class CDNet(nn.Module):
    def __init__(self,input_nbr=6,label_nbr=2):
        super(CDNet,self).__init__()
        filters = 64
        self.conv1 = nn.Conv2d(input_nbr, filters,kernel_size=7,padding=3,stride=1)
        self.bn = nn.BatchNorm2d(filters)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.conv2 = nn.Conv2d(filters,filters,kernel_size=7,padding=3,stride=1)
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)
        self.final = nn.Conv2d(filters,label_nbr,kernel_size=1,stride=1)
        #self.sigmod = nn.Sigmoid(dim=1)
        
        
    def forward(self,x1,x2):
        x = torch.cat((x1,x2),1)
        
        x = self.pool(self.relu(self.bn(self.conv1(x))))
        x = self.pool(self.relu(self.bn(self.conv2(x))))
        x = self.pool(self.relu(self.bn(self.conv2(x))))
        x = self.pool(self.relu(self.bn(self.conv2(x))))
        
        x = self.relu(self.bn(self.conv2(self.up(x))))
        x = self.relu(self.bn(self.conv2(self.up(x))))
        x = self.relu(self.bn(self.conv2(self.up(x))))
        x = self.relu(self.bn(self.conv2(self.up(x))))
        
        x = self.final(x)
        x = torch.sigmoid(x)
        #print(x.shape)
        
        return x
if __name__ == '__main__':
    from torchsummary import summary
    model = CDNet(input_nbr=6, label_nbr=2)
    summary(model,input_size=[(3,256,256),(3,256,256)],batch_size = 1, device="cpu")
