import captum
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

from net.convnet import ConvNet
from config import device 

BATCH = 128
transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5,0.5))])
trainset = torchvision.datasets.CIFAR10(root='./datasets', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH, shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./datasets', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH, shuffle=False)
model = ConvNet()
model.load_state_dict(torch.load('pth/conv.pth'))

dataiter = iter(test_loader)
images, labels = dataiter.next()
outputs = model(images)
_, predicted = torch.max(outputs, 1)
ind = 3
inputs = images[ind].unsqueeze(0)
inputs.requires_grad = True

# REF: https://www.zhihu.com/question/68384370/answer/751212803
cnt_conv = 0
cnt_relu = 0
def viz_conv(module, inputs):
    global cnt_conv
    cnt_conv += 1
    if cnt_conv > 8:
        return
    x = inputs[0][0]
    x = x.permute(1,2,0).detach().numpy()

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(x[:,:,i],cmap='gray')
        plt.axis('off')
    plt.savefig('dump/' + str(cnt_conv) + 'conv_act.png')

def viz_relu(module, inputs):
    global cnt_relu
    cnt_relu += 1
    if cnt_relu > 5:
        return
    x = inputs[0][0]
    x = x.permute(1,2,0).detach().numpy()
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(x[:,:,i],cmap='gray')
        plt.axis('off')
    plt.savefig('dump/' + str(cnt_relu) + 'relu_act.png')

for name, m in model.named_modules():
    if isinstance(m, torch.nn.Conv2d):
        m.register_forward_pre_hook(viz_conv)
    if isinstance(m, torch.nn.ReLU):
        m.register_forward_pre_hook(viz_relu)

model(inputs)

