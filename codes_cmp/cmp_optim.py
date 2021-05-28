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
LR = 3e-4
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
    acc_adam = []
    loss_adam = []
    acc_ada = []
    loss_ada = []
    acc_rms = []
    loss_rms = []
    acc_mom = []
    loss_mom = []
    acc_sgd = []
    loss_sgd = []

    model = ConvNet().to(device)
    optimizier = optim.Adam(model.parameters(),lr=LR)
    critirion = nn.CrossEntropyLoss()
    for epoch in range(20):
        loss = train(trainloader)
        testacc = test(testloader)
        acc_adam.append(testacc)
        loss_adam.append(loss)

        if epoch % 1 == 0:
            print('epoch',epoch,'loss',loss,'acc',testacc)
            #torch.save(model.state_dict(),model_path)

    model = ConvNet().to(device)
    optimizier = optim.SGD(model.parameters(),lr=LR)
    critirion = nn.CrossEntropyLoss()
    for epoch in range(20):
        loss = train(trainloader)
        testacc = test(testloader)
        acc_sgd.append(testacc)
        loss_sgd.append(loss)

        if epoch % 1 == 0:
            print('epoch',epoch,'loss',loss,'acc',testacc)
            #torch.save(model.state_dict(),model_path)

    model = ConvNet().to(device)
    optimizier = optim.SGD(model.parameters(),lr=LR,momentum=0.9)
    critirion = nn.CrossEntropyLoss()
    for epoch in range(20):
        loss = train(trainloader)
        testacc = test(testloader)
        acc_mom.append(testacc)
        loss_mom.append(loss)

        if epoch % 1 == 0:
            print('epoch',epoch,'loss',loss,'acc',testacc)
            #torch.save(model.state_dict(),model_path)

    model = ConvNet().to(device)
    optimizier = optim.Adagrad(model.parameters(),lr=LR)
    critirion = nn.CrossEntropyLoss()
    for epoch in range(20):
        loss = train(trainloader)
        testacc = test(testloader)
        acc_ada.append(testacc)
        loss_ada.append(loss)

        if epoch % 1 == 0:
            print('epoch',epoch,'loss',loss,'acc',testacc)
            #torch.save(model.state_dict(),model_path)
    
    model = ConvNet().to(device)
    optimizier = optim.RMSprop(model.parameters(),lr=LR)
    critirion = nn.CrossEntropyLoss()
    for epoch in range(20):
        loss = train(trainloader)
        testacc = test(testloader)
        acc_rms.append(testacc)
        loss_rms.append(loss)

        if epoch % 1 == 0:
            print('epoch',epoch,'loss',loss,'acc',testacc)
            #torch.save(model.state_dict(),model_path)
    
    plt.subplot(1,2,1)
    plt.plot(loss_adam,label='Adam')
    plt.plot(loss_ada,label='Adagrad')
    plt.plot(loss_sgd,label='SGD')
    plt.plot(loss_mom,label='SGD+momentum')
    plt.plot(loss_rms,label='RMSprop')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(acc_adam,label='Adam')
    plt.plot(acc_ada,label='Adagrad')
    plt.plot(acc_sgd,label='SGD')
    plt.plot(acc_mom,label='SGD+momentum')
    plt.plot(acc_rms,label='RMSprop')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()

    plt.savefig('dump/compare_optim.png')

