import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from net.convnet import ConvNet, ConvNet_Sigmoid, ConvNet_Tanh, ConvNet_Elu
from config import device
import matplotlib.pyplot as plt
import tqdm 

BATCH = 128
LR = 1e-3
transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5,0.5))])
trainset = torchvision.datasets.CIFAR10(root='./datasets', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH, shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./datasets', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH, shuffle=False)

def cls_train(dataloader):
    model.train()
    tot_loss = 0
    tot_num = 0
    for i,data in tqdm.tqdm(enumerate(dataloader)):
        if i>100:
            break
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
        if i>100:
            break
        x,y = data
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        pred = logits.argmax(1)
        correct += torch.sum(pred==y).item()
        num += x.shape[0]
    acc = correct / num
    return acc 

def reg_train(dataloader):
    model.train()
    tot_loss = 0
    tot_num = 0
    for i,data in tqdm.tqdm(enumerate(dataloader)):
        if i>100:
            break
        optimizier.zero_grad()
        x,y = data
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        labels = torch.zeros_like(logits)
        n_batch = labels.shape[0]
        for b in range(n_batch):
            labels[b,y[b]] = 1
        loss = critirion(logits,labels)
        loss.backward()
        optimizier.step()

        tot_loss += loss.item()
        tot_num += x.shape[0]
    return tot_loss / tot_num

if __name__ == '__main__':
    acc_cross = []
    loss_cross = []
    acc_mae = []
    loss_mae = []
    acc_mse = []
    loss_mse = []
    acc_sm = []
    loss_sm = []

    model = ConvNet().to(device)
    optimizier = optim.Adam(model.parameters(),lr=LR)
    critirion = nn.CrossEntropyLoss()
    for epoch in range(20):
        loss = cls_train(trainloader)
        testacc = test(testloader)
        acc_cross.append(testacc)
        loss_cross.append(loss)

        if epoch % 1 == 0:
            print('epoch',epoch,'loss',loss,'acc',testacc)
            #torch.save(model.state_dict(),model_path)

    model = ConvNet().to(device)
    optimizier = optim.Adam(model.parameters(),lr=LR)
    critirion = nn.L1Loss()
    for epoch in range(20):
        loss = reg_train(trainloader)
        testacc = test(testloader)
        acc_mae.append(testacc)
        loss_mae.append(loss)

        if epoch % 1 == 0:
            print('epoch',epoch,'loss',loss,'acc',testacc)
            #torch.save(model.state_dict(),model_path)

    model = ConvNet().to(device)
    optimizier = optim.Adam(model.parameters(),lr=LR)
    critirion = nn.MSELoss()
    for epoch in range(20):
        loss = reg_train(trainloader)
        testacc = test(testloader)
        acc_mse.append(testacc)
        loss_mse.append(loss)

        if epoch % 1 == 0:
            print('epoch',epoch,'loss',loss,'acc',testacc)
            #torch.save(model.state_dict(),model_path)

    model = ConvNet().to(device)
    optimizier = optim.Adam(model.parameters(),lr=LR)
    critirion = nn.SmoothL1Loss()
    for epoch in range(20):
        loss = reg_train(trainloader)
        testacc = test(testloader)
        acc_sm.append(testacc)
        loss_sm.append(loss)

        if epoch % 1 == 0:
            print('epoch',epoch,'loss',loss,'acc',testacc)
            #torch.save(model.state_dict(),model_path)

    plt.subplot(1,2,1)
    plt.plot(loss_cross,label='CrossEntropy')
    plt.plot(loss_mae,label='MAE')
    plt.plot(loss_mse,label='MSE')
    plt.plot(loss_sm,label='SmoothL1')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(acc_cross,label='CrossEntropy')
    plt.plot(acc_mae,label='MAE')
    plt.plot(acc_mse,label='MSE')
    plt.plot(acc_sm,label='SmoothL1')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()

    plt.savefig('dump/compare_critirion.png')

