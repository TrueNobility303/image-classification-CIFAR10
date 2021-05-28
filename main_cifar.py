import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from net.convnet import ConvNet
from net.resnet import ResNet, Attention_ResNet
from net.fcn import FCN
from net.vgg import VGG_A_BatchNorm
from config import device
import matplotlib.pyplot as plt
import tqdm 
from optim.sgd import mySGD

BATCH = 512
transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5,0.5))])
trainset = torchvision.datasets.CIFAR10(root='./datasets', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH, shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./datasets', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH, shuffle=False)

model = ResNet().to(device)
model_path = 'pth/resnet517.pth'
exp_path = 'exp/resnet517.png'

#lr=3e-4
#optimizier = optim.Adam(model.parameters(),lr=1e-3)
optimizier = mySGD(model.parameters(),lr=1e-2)
critirion = nn.CrossEntropyLoss()

#使用convnet在10轮次，
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
    train_accs = []
    test_accs = []
    for epoch in range(20):
        loss = train(trainloader)
        trainacc = test(trainloader)
        testacc = test(testloader)
        train_accs.append(trainacc)
        test_accs.append(testacc)

        if epoch % 1 == 0:
            print('epoch',epoch,'loss',loss,'acc',testacc)
            torch.save(model.state_dict(),model_path)
    plt.figure()
    plt.plot(train_accs)
    plt.plot(test_accs)
    plt.legend(['train','test'])
    plt.savefig(exp_path)




