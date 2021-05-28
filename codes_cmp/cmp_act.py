import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from net.convnet import ConvNet, ConvNet_Sigmoid, ConvNet_Tanh, ConvNet_Elu

from config import device
import matplotlib.pyplot as plt
import tqdm 
from vision.alexnet import AlexNet

BATCH = 128
LR = 1e-3
transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5,0.5))])
trainset = torchvision.datasets.CIFAR10(root='./datasets', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH, shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./datasets', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH, shuffle=False)

def train(dataloader):
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

if __name__ == '__main__':
    acc_relu = []
    loss_relu = []
    acc_elu = []
    loss_elu = []
    acc_sigmoid = []
    loss_simoid = []
    acc_tanh = []
    loss_tanh = []

    model = ConvNet().to(device)
    optimizier = optim.Adam(model.parameters(),lr=LR)
    critirion = nn.CrossEntropyLoss()
    for epoch in range(20):
        loss = train(trainloader)
        testacc = test(testloader)
        acc_relu.append(testacc)
        loss_relu.append(loss)

        if epoch % 1 == 0:
            print('epoch',epoch,'loss',loss,'acc',testacc)
            #torch.save(model.state_dict(),model_path)


    model = ConvNet_Sigmoid().to(device)
    optimizier = optim.Adam(model.parameters(),lr=LR)
    critirion = nn.CrossEntropyLoss()
    for epoch in range(20):
        loss = train(trainloader)
        testacc = test(testloader)
        acc_sigmoid.append(testacc)
        loss_simoid.append(loss)

        if epoch % 1 == 0:
            print('epoch',epoch,'loss',loss,'acc',testacc)
            #torch.save(model.state_dict(),model_path)
    
    model = ConvNet_Tanh().to(device)
    optimizier = optim.Adam(model.parameters(),lr=LR)
    critirion = nn.CrossEntropyLoss()
    for epoch in range(20):
        loss = train(trainloader)
        testacc = test(testloader)
        acc_tanh.append(testacc)
        loss_tanh.append(loss)

        if epoch % 1 == 0:
            print('epoch',epoch,'loss',loss,'acc',testacc)
            #torch.save(model.state_dict(),model_path)
    
    model = ConvNet_Elu().to(device)
    optimizier = optim.Adam(model.parameters(),lr=LR)
    critirion = nn.CrossEntropyLoss()
    for epoch in range(20):
        loss = train(trainloader)
        testacc = test(testloader)
        acc_elu.append(testacc)
        loss_elu.append(loss)

        if epoch % 1 == 0:
            print('epoch',epoch,'loss',loss,'acc',testacc)
            #torch.save(model.state_dict(),model_path)

    plt.subplot(1,2,1)
    plt.plot(loss_relu,label='relu')
    plt.plot(loss_elu,label='elu')
    plt.plot(loss_simoid,label='sigmoid')
    plt.plot(loss_tanh,label='tanh')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(acc_relu,label='relu')
    plt.plot(acc_elu,label='elu')
    plt.plot(acc_sigmoid,label='sigmoid')
    plt.plot(acc_tanh,label='tanh')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()

    plt.savefig('dump/compare_activate.png')

