import matplotlib
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm
from config import device 
from net.vgg import VGG_A, VGG_A_BatchNorm
import torchvision
import torchvision.transforms as transforms

matplotlib.use('Agg')
BATCH = 128
transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5,0.5))])
trainset = torchvision.datasets.CIFAR10(root='/datasets/CIFAR10', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH, shuffle=True)
testset = torchvision.datasets.CIFAR10(root='/datasets/CIFAR10', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH, shuffle=False)

def set_random_seeds(seed_value=0):
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random_seeds(42)
lrs = [6e-4,8e-4,1e-3]

#取20轮次，分为4个学习率，完整训练每一单元时间大约需要10分钟每次训练，总计需要80分钟左右，取部分数据集大约在半小时内完成

#更改epoch
def train(model, optimizer, criterion, train_loader, val_loader, epochs_n=20):
    #记录每一轮的准确率
    train_accs = []
    val_accs = []

    #记录每一次迭代过程的损失和梯度
    losses = []
    grads = []
    weights = []
    icses = []

    for epoch in tqdm(range(epochs_n), unit='epoch'):
        model.train()

        tot_num = 0
        correct = 0
        for i,data in enumerate(train_loader):
            if i>100:
                break
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            grad = model.classifier[-1].weight.grad.clone()
            wei = model.classifier[-1].weight.clone()

            optimizer.step()

            new_grad = model.classifier[-1].weight.grad.clone()
            ICS = torch.linalg.norm(new_grad-grad,2)
            
            losses.append(loss.item())
            grads.append(grad.cpu().detach().numpy())
            weights.append(wei.cpu().detach().numpy())
            icses.append(ICS)

            pred = logits.argmax(1)
            correct += torch.sum(pred==y).item()
            tot_num += x.shape[0]
        
        train_accs.append(correct / tot_num)

        model.eval()
        tot_num = 0
        correct = 0
        for i,data in enumerate(val_loader):
            if i>100:
                break
            x, y = data
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = logits.argmax(1)
            
            correct += torch.sum(pred==y).item()
            tot_num += x.size(0)

        train_accs.append(correct / tot_num)
    
    return losses, grads, train_accs, val_accs, weights, icses

vgg_loss_lst = []
vgg_grad_lst = []
vgg_train_lst = []
vgg_val_lst = []
vgg_wei_lst = []
vgg_ics_lst = []

for lr in lrs:
    set_random_seeds(42)
    model = VGG_A().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    criterion = nn.CrossEntropyLoss()
    losses, grads, train_accs, val_accs,weights,icses = train(model, optimizer, criterion, train_loader, test_loader)
    vgg_loss_lst.append(losses)
    vgg_grad_lst.append(grads)
    vgg_train_lst.append(train_accs)
    vgg_val_lst.append(val_accs)
    vgg_wei_lst.append(weights)
    vgg_ics_lst.append(icses)

vgg_bn_loss_lst = []
vgg_bn_grad_lst = []
vgg_bn_train_lst = []
vgg_bn_val_lst = []
vgg_bn_wei_lst = []
vgg_bn_ics_lst = []

for lr in lrs:
    set_random_seeds(42)
    model = VGG_A_BatchNorm().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    criterion = nn.CrossEntropyLoss()
    losses, grads, train_accs, val_accs,weights,icses = train(model, optimizer, criterion, train_loader, test_loader)
    vgg_bn_loss_lst.append(losses)
    vgg_bn_grad_lst.append(grads)
    vgg_bn_train_lst.append(train_accs)
    vgg_bn_val_lst.append(val_accs)
    vgg_bn_wei_lst.append(weights)
    vgg_bn_ics_lst.append(icses)

vgg_bn_grad_lst = np.stack(vgg_bn_grad_lst)
vgg_bn_loss_lst = np.stack(vgg_bn_loss_lst)
vgg_bn_train_lst = np.stack(vgg_bn_train_lst)
vgg_bn_val_lst = np.stack(vgg_bn_val_lst)
vgg_grad_lst = np.stack(vgg_grad_lst)
vgg_loss_lst = np.stack(vgg_loss_lst)
vgg_train_lst = np.stack(vgg_train_lst)
vgg_val_lst = np.stack(vgg_val_lst)
vgg_bn_wei_lst = np.stack(vgg_bn_wei_lst)
vgg_wei_lst = np.stack(vgg_wei_lst)
vgg_ics_lst = np.stack(vgg_ics_lst)
vgg_bn_ics_lst = np.stack(vgg_bn_ics_lst)

savepath = 'landscape/'
np.save(savepath + 'vgg_bn_grad_lst.npy',vgg_bn_grad_lst,allow_pickle=True)
np.save(savepath + 'vgg_bn_loss_lst.npy',vgg_bn_loss_lst,allow_pickle=True)
np.save(savepath+'vgg_bn_train_lst.npy',vgg_bn_train_lst,allow_pickle=True)
np.save(savepath + 'vgg_bn_val_lst.npy',vgg_bn_val_lst,allow_pickle=True)
np.save(savepath + 'vgg_bn_wei_lst.npy',vgg_bn_wei_lst,allow_pickle=True)
np.save(savepath + 'vgg_bn_ics_lst.npy',vgg_bn_ics_lst,allow_pickle=True)

np.save(savepath + 'vgg_grad_lst.npy',vgg_grad_lst,allow_pickle=True)
np.save(savepath + 'vgg_loss_lst.npy',vgg_loss_lst,allow_pickle=True)
np.save(savepath+'vgg_train_lst.npy',vgg_train_lst,allow_pickle=True)
np.save(savepath + 'vgg_val_lst.npy',vgg_val_lst, allow_pickle=True)
np.save(savepath + 'vgg_wei_lst.npy',vgg_wei_lst,allow_pickle=True)
np.save(savepath + 'vgg_ics_lst.npy',vgg_ics_lst,allow_pickle=True)
