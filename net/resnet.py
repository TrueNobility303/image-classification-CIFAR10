import torch.nn as nn
import torch.nn.functional as F
import torch 
import torchvision
from torchvision import models
from torchsummary import summary
from config import device

class CBR(nn.Module):
    def __init__(self,in_channels,out_channels,k_size,stride,padding,inplace=True):
        super().__init__()
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        conv = nn.Conv2d(in_channels, out_channels, k_size, stride=stride, padding=padding)
        self.cbr = nn.Sequential(conv, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=inplace))
    
    def forward(self,x):
        return self.cbr(x)

class CR(nn.Module):
    def __init__(self,in_channels,out_channels,k_size,stride,padding,inplace=True):
        super().__init__()
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        conv = nn.Conv2d(in_channels, out_channels, k_size, stride=stride, padding=padding)
        self.cr = nn.Sequential(conv, nn.ReLU(inplace=inplace))
    
    def forward(self,x):
        return self.cr(x)

class CB(nn.Module):
    def __init__(self,in_channels,out_channels,k_size,stride,padding):
        super().__init__()
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        conv = nn.Conv2d(in_channels,out_channels,k_size,stride,padding)
        self.cb = nn.Sequential(conv, nn.BatchNorm2d(out_channels))
    def forward(self,x):
        return self.cb(x)
        
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,inplace=True):
        super().__init__()
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        self.cbr = CBR(in_channels, out_channels, 3, stride, 1)
        self.cb = CB(out_channels, out_channels, 3, 1, 1)
        #在CB R之间引入残差单元
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        out = self.cbr(x)
        residual = out
        out = self.cb(out)
        #print(out.shape,residual.shape)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self,n_classes=10):
        super().__init__()
        self.n_classes = n_classes

        self.conv_block1 = nn.Sequential(
            ResBlock(3,64),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.conv_block2 = nn.Sequential(
            ResBlock(64,128),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.conv_block3 = nn.Sequential(
            ResBlock(128,256),
            ResBlock(256,256),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(4*4*256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, self.n_classes),
        )

    def forward(self,x):
        n_batch = x.shape[0]
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.view(n_batch,-1)
        y = self.classifier(x)
        return y

#REF: https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Attention_ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,inplace=True):
        super().__init__()
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        self.cbr = CBR(in_channels, out_channels, 3, stride, 1)
        self.cb = CB(out_channels, out_channels, 3, 1, 1)
        #在CB R之间引入残差单元
        self.relu = nn.ReLU(inplace=inplace)
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        out = self.cbr(x)
        residual = out
        out = self.cb(out)

        #注意力机制
        out = self.ca(out) * out
        out = self.sa(out) * out

        #print(out.shape,residual.shape)
        out += residual
        out = self.relu(out)
        return out

class Attention_ResNet(nn.Module):
    def __init__(self,n_classes=10):
        super().__init__()
        self.n_classes = n_classes

        self.conv_block1 = nn.Sequential(
            Attention_ResBlock(3,64),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.conv_block2 = nn.Sequential(
            Attention_ResBlock(64,128),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.conv_block3 = nn.Sequential(
            Attention_ResBlock(128,256),
            Attention_ResBlock(256,256),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(4*4*256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, self.n_classes),
        )

    def forward(self,x):
        n_batch = x.shape[0]
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.view(n_batch,-1)
        y = self.classifier(x)
        return y
