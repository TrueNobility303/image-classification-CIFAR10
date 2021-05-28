
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
#对比SAM和普通的SGD的效果

#from cmp_sam 


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

    model = WideResNet(depth=16, width_factor=8, dropout=0, in_channels=3, labels=10)
    base_optimizer = torch.optim.SGD
    optimizier = SAM(model.parameters(), base_optimizer, rho=0.5, adaptive=True, lr=LR, momentum=0.9, weight_decay=DECAY)
    scheduler = StepLR(optimizier, LR, EPOCH)

    lr_of_epoches = []

    for epoch in range(EPOCH):
        
        scheduler(epoch)
        learning_rate = scheduler.lr()
        #print(learning_rate)
        lr_of_epoches.append(learning_rate)

    plt.plot(lr_of_epoches)
    plt.xlabel('epoch')
    plt.ylabel('learning rate')
    #plt.legend()
    plt.savefig('exp/show_stepLR.png')
    
    #model_path = 'pth/wrn_sam.pth'
    #torch.save(model.state_dict(),model_path)

