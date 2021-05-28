import captum
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import torch
import torchvision
import torchvision.transforms as transforms

from net.convnet import ConvNet
from config import device 
from torch import nn
import torch.nn.functional as F 

BATCH = 128
transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5,0.5))])
trainset = torchvision.datasets.CIFAR10(root='./datasets', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH, shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./datasets', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH, shuffle=False)
model = ConvNet()
model.load_state_dict(torch.load('pth/conv.pth'))

#REF: https://mathpretty.com/10475.html
class SaveFeatures():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        # self.features = output.clone().detach().requires_grad_(True)
        self.features = output.clone()
    def close(self):
        self.hook.remove()

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(*list(model.children())[:-2]) 

    def forward(self,x):
        b,c,h,w = x.shape
        y = self.net(x)
        y = F.interpolate(x,size=(h,w))
        return y

if __name__ =='__main__':
    model.eval()
    net = Net().to(device)
    net.eval()
    
    img = torch.rand((1,3,32,32)).to(device)
    img = (img-0.5) / 0.5 
    optimizer = torch.optim.Adam([img.requires_grad_()], lr=0.01)
    for e in range(2000):
        optimizer.zero_grad()
        img = net(img)
        loss = -img.norm()
        loss.backward()
        optimizer.step()

    img = img*0.5 +0.5
    img = img.squeeze(0)
    img = img.permute(1,2,0)
            
    gimg = np.zeros((32,32)) 
    img = img.detach().cpu().numpy()
    gimg[:,:] = img[:,:,0] * 0.299 + img[:,:,1] * 0.587 + img[:,:,2] * 0.114
    plt.imshow(gimg,cmap='gray')
    plt.axis('off')
    plt.savefig('dump/dreaming.png')
