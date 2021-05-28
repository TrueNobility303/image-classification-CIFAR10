from optim.sam import SAM
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from net.convnet import ConvNet, ConvNet_Sigmoid, ConvNet_Tanh, ConvNet_Elu
from config import device
import matplotlib.pyplot as plt
import tqdm 
from optim.adam import myAdam
from optim.adamw import myAdamW
from optim.sgd import mySGD 
from optim.sgdw import mySGDW
from net.resnet import ResNet
from net.wrn import WideResNet,smooth_crossentropy
import random


#复现WRN+SAM，acc 97.2%,并且对比SAM和普通的SGD的效果

BATCH = 128
LR = 0.1
DECAY = 5e-4
EPOCH = 200 
#1800

def initialize(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

initialize(42)

#from 
class StepLR:
    def __init__(self, optimizer, learning_rate: float, total_epochs: int):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.base = learning_rate

    def __call__(self, epoch):
        if epoch < self.total_epochs * 3/10:
            lr = self.base
        elif epoch < self.total_epochs * 6/10:
            lr = self.base * 0.2
        elif epoch < self.total_epochs * 8/10:
            lr = self.base * 0.2 ** 2
        else:
            lr = self.base * 0.2 ** 3

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

#数据扩增
class Cutout:
    def __init__(self, size=16, p=0.5):
        self.size = size
        self.half_size = size // 2
        self.p = p

    def __call__(self, image):
        if torch.rand([1]).item() > self.p:
            return image

        left = torch.randint(-self.half_size, image.size(1) - self.half_size, [1]).item()
        top = torch.randint(-self.half_size, image.size(2) - self.half_size, [1]).item()
        right = min(image.size(1), left + self.size)
        bottom = min(image.size(2), top + self.size)

        image[:, max(0, left): right, max(0, top): bottom] = 0
        return image

mean = [0.4913997551666284, 0.48215855929893703, 0.4465309133731618]
std = [0.24703225141799082, 0.24348516474564, 0.26158783926049628]

train_transform = transforms.Compose([
            torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            Cutout()
        ])

test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

trainset = torchvision.datasets.CIFAR10(root='/datasets/CIFAR10', train=True, download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH, shuffle=True)
testset = torchvision.datasets.CIFAR10(root='/datasets/CIFAR10', train=False, download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH, shuffle=False)

def train(dataloader,train_sam=False):
    model.train()
    tot_loss = 0
    tot_num = 0
    for i,data in enumerate(dataloader):
        x,y = data
        x = x.to(device)
        y = y.to(device)
        logits = model(x)

        #非SAM
        if train_sam is False:
            optimizier.zero_grad()
            loss = smooth_crossentropy(logits,y).mean()
            loss.backward()
            optimizier.step()

        else:
            loss = smooth_crossentropy(logits,y).mean()
            loss.backward()
            optimizier.first_step(zero_grad=True)

            # second forward-backward step
            smooth_crossentropy(model(x), y).mean().backward()
            optimizier.second_step(zero_grad=True)

        tot_loss += loss.item()
        tot_num += x.shape[0]

    return tot_loss / tot_num

def test(dataloader):
    model.eval()
    correct = 0
    num = 0
    for i,data in enumerate(dataloader):
        x,y = data
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        pred = logits.argmax(1)
        correct += torch.sum(pred==y).item()
        num += x.shape[0]
    acc = correct / num
    return acc 

if __name__ == '__main__':
    acc_sgd = []
    loss_sgd = []
    acc_sgd_sam = []
    loss_sgd_sam = []

    model = WideResNet(depth=16, width_factor=8, dropout=0, in_channels=3, labels=10).to(device)
    base_optimizer = torch.optim.SGD
    optimizier = SAM(model.parameters(), base_optimizer, rho=0.5, adaptive=True, lr=LR, momentum=0.9, weight_decay=DECAY)
    scheduler = StepLR(optimizier, LR, EPOCH)
    for epoch in range(EPOCH):
        #训练sam大叔优化器
        loss = train(trainloader,train_sam=True)
        testacc = test(testloader)
        acc_sgd_sam.append(testacc)
        loss_sgd_sam.append(loss)

        if epoch % 1 == 0:
            print('epoch',epoch,'loss',loss,'acc',testacc)
        
        scheduler(epoch)
    
    model_path = 'pth/wrn_sam.pth'
    torch.save(model.state_dict(),model_path)

    model = WideResNet(depth=16, width_factor=8, dropout=0, in_channels=3, labels=10).to(device)
    
    optimizier = torch.optim.SGD(model.parameters(),lr=LR,momentum=0.9,weight_decay=DECAY)
    scheduler = StepLR(optimizier, LR, EPOCH)
    for epoch in range(EPOCH):
        loss = train(trainloader)
        testacc = test(testloader)
        acc_sgd.append(testacc)
        loss_sgd.append(loss)

        if epoch % 1 == 0:
            print('epoch',epoch,'loss',loss,'acc',testacc)
        scheduler(epoch)
    
    model_path = 'pth/wrn_sgd.pth'
    torch.save(model.state_dict(),model_path)
    
    plt.subplot(1,2,1)
    plt.plot(loss_sgd,label='SGD')
    plt.plot(loss_sgd_sam,label='SGD_SAM')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(acc_sgd,label='SGD')
    plt.plot(acc_sgd_sam,label='SGD_SAM')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()

    plt.savefig('dump/compare_sam.png')

