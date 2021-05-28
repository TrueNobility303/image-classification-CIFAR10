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
from optim.kfac import KFACOptimizer

#对比KFAC_SGD和SGD的效果

#用KFAC进行训练

BATCH = 512
LR = 1e-3
DECAY = 0

transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5,0.5))])
trainset = torchvision.datasets.CIFAR10(root='/datasets/CIFAR10', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH, shuffle=True)
testset = torchvision.datasets.CIFAR10(root='/datasets/CIFAR10', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH, shuffle=False)

def train(dataloader):
    model.train()
    tot_loss = 0
    tot_num = 0
    for i,data in tqdm.tqdm(enumerate(dataloader)):
        #if i>100:
        #    break
        optimizier.zero_grad()
        x,y = data
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = critirion(logits,y)
        loss.backward()
        optimizier.step()

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
    acc_adam = []
    loss_adam = []
    acc_adamw = []
    loss_adamw = []
    
    acc_sgd = []
    loss_sgd = []
    acc_sgdw = []
    loss_sgdw = []

    model = ResNet().to(device)
    optimizier = KFACOptimizer(model,lr=LR, weight_decay=DECAY)
    
    #KFAC的超参数
    critirion = nn.CrossEntropyLoss()
    for epoch in range(20):
        #KFAC累计统计量
        optimizier.acc_stats = True 
        loss = train(trainloader)
        testacc = test(testloader)
        acc_adamw.append(testacc)
        loss_adamw.append(loss)
        optimizier.acc_stats = False

        if epoch % 1 == 0:
            print('epoch',epoch,'loss',loss,'acc',testacc)
            #torch.save(model.state_dict(),model_path)
    
    model = ResNet().to(device)
    optimizier = torch.optim.Adam(model.parameters(),lr=LR,weight_decay=DECAY)
    critirion = nn.CrossEntropyLoss()
    for epoch in range(20):
        loss = train(trainloader)
        testacc = test(testloader)
        acc_adam.append(testacc)
        loss_adam.append(loss)

        if epoch % 1 == 0:
            print('epoch',epoch,'loss',loss,'acc',testacc)
            #torch.save(model.state_dict(),model_path)

    plt.subplot(1,2,1)
    plt.plot(loss_adam,label='Adam')
    plt.plot(loss_adamw,label='Adam_KFAC')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(acc_adam,label='Adam')
    plt.plot(acc_adamw,label='Adam_KFAC')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()

    plt.savefig('dump/compare_kfac.png')

