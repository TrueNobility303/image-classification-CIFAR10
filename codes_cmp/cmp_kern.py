import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from net.convnet import ConvNet, ConvNet_Big, ConvNet_Bigger
from config import device
import matplotlib.pyplot as plt
import tqdm 
from torchsummary import summary 

BATCH = 128
LR = 1e-3
transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5,0.5))])
trainset = torchvision.datasets.CIFAR10(root='./datasets', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH, shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./datasets', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH, shuffle=False)

model_sm = ConvNet().to(device)
model_bg = ConvNet_Big().to(device)
model_bger = ConvNet_Bigger().to(device)

def train(dataloader):
    model.train()
    tot_loss = 0
    tot_num = 0
    for i,data in tqdm.tqdm(enumerate(dataloader)):
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

    acc_sm = []
    loss_sm = []
    acc_bg = []
    loss_bg = []
    acc_bger = []
    loss_bger = []

    inputs = (3,32,32)
    summary(model_sm,inputs)
    summary(model_bg,inputs)
    summary(model_bger,inputs)
    model = model_sm
    optimizier = optim.Adam(model.parameters(),lr=LR)
    critirion = nn.CrossEntropyLoss()
    for epoch in range(20):
        loss = train(trainloader)
        testacc = test(testloader)
        acc_sm.append(testacc)
        loss_sm.append(loss)

        if epoch % 1 == 0:
            print('epoch',epoch,'loss',loss,'acc',testacc)
            #torch.save(model.state_dict(),model_path)

    model = model_bg
    optimizier = optim.Adam(model.parameters(),lr=LR)
    critirion = nn.CrossEntropyLoss()
    for epoch in range(20):
        loss = train(trainloader)
        testacc = test(testloader)
        acc_bg.append(testacc)
        loss_bg.append(loss)

        if epoch % 1 == 0:
            print('epoch',epoch,'loss',loss,'acc',testacc)

    model = model_bger
    optimizier = optim.Adam(model.parameters(),lr=LR)
    critirion = nn.CrossEntropyLoss()
    for epoch in range(20):
        loss = train(trainloader)
        testacc = test(testloader)
        acc_bger.append(testacc)
        loss_bger.append(loss)

        #if epoch % 1 == 0:
        #    print('epoch',epoch,'loss',loss,'acc',testacc)

    plt.subplot(1,2,1)
    plt.plot(loss_sm,label='3x3 kernel')
    plt.plot(loss_bg,label='5x5 kernel')
    plt.plot(loss_bger,label='7x7 kernel')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(acc_sm,label='3x3 kernel')
    plt.plot(acc_bg,label='5x5 kernel')
    plt.plot(acc_bger,label='7x7 kernel')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()

    plt.savefig('dump/compare_kern.png')

