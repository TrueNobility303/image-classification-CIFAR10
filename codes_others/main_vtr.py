import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from net.convnet import ConvNet
from net.resnet import ResNet, Attention_ResNet
from net.fcn import FCN
from net.vgg import VGG_A_BatchNorm
from net.vtr import VisionTransformer
from config import device
import matplotlib.pyplot as plt
import tqdm 
from torch.optim.lr_scheduler import StepLR

#定义batch训练
BATCH = 64
transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5,0.5))])
trainset = torchvision.datasets.CIFAR10(root='/datasets/CIFAR10', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH, shuffle=True)
testset = torchvision.datasets.CIFAR10(root='/datasets/CIFAR10', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH, shuffle=False)

model = VisionTransformer().to(device)
model_path = 'pth/vtr.pth'
exp_path = 'exp/vtr.png'

#lr=3e-4
#参数 finetune in CIFAR10 10000it {0.001, 0.003, 0.01, 0.03}

optimizier = optim.AdamW(model.parameters(),lr=1e-3)
scheduler = StepLR(optimizier, step_size=5, gamma=0.5)
#optimizier = mySGD(model.parameters(),lr=1e-2)
critirion = nn.CrossEntropyLoss()

#使用convnet在10轮次，
def train(dataloader):
    model.train()
    tot_loss = 0
    tot_num = 0
    for i,data in enumerate(dataloader):
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
        #if i>100:
        #    break
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
    losses = []
    EPOCH = 20 
    for epoch in range(EPOCH):
        loss = train(trainloader)
        trainacc = test(trainloader)
        testacc = test(testloader)
        train_accs.append(trainacc)
        test_accs.append(testacc)
        losses.append(loss)

        scheduler.step()

        if epoch % 1 == 0:
            print('epoch',epoch,'loss',loss,'train',trainacc, 'test',testacc)
            torch.save(model.state_dict(),model_path)

    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(train_accs)
    plt.plot(test_accs)
    plt.legend(['train','test'])
    plt.title('accuracy')

    plt.subplot(1,2,2)
    plt.plot(loss)
    plt.title('loss')
    plt.savefig(exp_path)




