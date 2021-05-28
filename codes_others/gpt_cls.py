import numpy as np
from torch.nn.modules.activation import ReLU
from torch.nn.modules.linear import Linear
from torch.optim import optimizer
from torch.utils.data import dataloader, dataset
import torchvision
import torch
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from net.gpt import GPT,GPTConfig, GPT_CLS, ConvNet, GPT_TR
from torch.utils.data.dataloader import DataLoader
from config import device
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.nn import functional as F

#REF: https://github.com/karpathy/minGPT/blob/master/play_image.ipynb

#截取部分数据集
class PartialDataset(Dataset):
    def __init__(self, dataset, n_items):
        self.dataset = dataset
        self.n_items = n_items

    def __getitem__(self,index):
        return self.dataset.__getitem__(index)

    def __len__(self):
        return min(self.n_items, len(self.dataset))

def set_random_seeds(seed_value=0):
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random_seeds(42)

BATCH = 20
#5it/s

transform = transforms.Compose( [transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='/datasets/CIFAR10', train=True, download=True)
testset = torchvision.datasets.CIFAR10(root='/datasets/CIFAR10', train=False, download=True)
trainset = PartialDataset(trainset,1000)
testset = PartialDataset(testset,500)

dataset = trainset
pluck_rgb = lambda x: torch.from_numpy(np.array(x)).view(32*32, 3)[torch.randperm(32*32)[:5], :]
px = torch.cat([pluck_rgb(x) for x, y in dataset], dim=0).float()

#在cifar上定义数据集
class ImageDataset(Dataset):
    def __init__(self, pt_dataset, clusters, perm=None):
        self.pt_dataset = pt_dataset
        self.clusters = clusters
        self.perm = torch.arange(32*32) if perm is None else perm
        #词汇量为聚类簇个数
        self.vocab_size = clusters.size(0)
        self.block_size = 32*32 - 1
        
    def __len__(self):
        return len(self.pt_dataset)

    def __getitem__(self, idx):
        x, y = self.pt_dataset[idx]
        x = torch.from_numpy(np.array(x)).view(-1, 3)
        x = x[self.perm].float() 
        a = ((x[:, None, :] - self.clusters[None, :, :])**2).sum(-1).argmin(1) 
        return a[:-1], a[1:], y

#使用KMeans聚类算法得到全局变量CLUSTER_DICT
def kmeans(x, ncluster, niter=10):
    N, D = x.size()
    #聚类中心
    c = x[torch.randperm(N)[:ncluster]] 
    for i in range(niter):
        a = ((x[:, None, :] - c[None, :, :])**2).sum(-1).argmin(1)
        c = torch.stack([x[a==k].mean(0) for k in range(ncluster)])
        nanix = torch.any(torch.isnan(c), dim=1)
        ndead = nanix.sum().item()
        c[nanix] = x[torch.randperm(N)[:ndead]] 
    return c

NCLUSTER = 512
with torch.no_grad():
    ClUSTER_DICT = kmeans(px, NCLUSTER)
print('finsh kmeans')

#按照0.8的比例划分
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
trainset, testset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_dataset = ImageDataset(trainset, ClUSTER_DICT)
test_dataset = ImageDataset(testset, ClUSTER_DICT)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH, shuffle=False)

#配置GPT模型，并且定义模型
mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                  embd_pdrop=0.5, resid_pdrop=0.5, attn_pdrop=0.5,
                  n_layer=4, n_head=2, n_embd=64)

model = GPT_TR(mconf).to(device)
optimizer = model.configure_optimizers(lr=3e-4,weight_decay=5e-4)

#使用卷积网络check
#model = ConvNet().to(device)
#optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
critirion = nn.CrossEntropyLoss()

def train(dataloader):
    model.train()
    tot_loss = 0
    tot_num = 0
    tot_correct = 0
    for i,data in tqdm.tqdm(enumerate(dataloader)):
        optimizer.zero_grad()
        x,y,label = data
        x = x.to(device)
        y = y.to(device)

        label = label.to(device)
        logits,_ = model(x)
        
        loss = critirion(logits,label) 
        loss.backward()
        optimizer.step()

        tot_loss += loss.item()
        tot_num += torch.sum(label>=0).item()

    return tot_loss / tot_num

def test(dataloader):
    model.eval()
    tot_num = 0
    tot_correct = 0
    for i,data in tqdm.tqdm(enumerate(dataloader)):

        x,y,label = data
        x = x.to(device)
        y = y.to(device)
        label = label.to(device)
        logits,_ = model(x)

        pred = logits.argmax(-1)
        tot_correct += torch.sum(pred==label).item() 
        tot_num += torch.sum(label>=0).item()

    return tot_correct / tot_num

if __name__ == '__main__':
    train_accs = []
    test_accs = []
    
    exp_path = 'exp/cls_gpt.png'
    model_path = 'pth/cls_gpt.pth'

    for epoch in range(20):
        loss = train(trainloader)
        test_acc = test(testloader)
        train_acc = test(trainloader)
        print('epoch',epoch,'loss',loss,'train',train_acc,'test',test_acc)
        test_accs.append(test_acc)
        train_accs.append(train_acc)
        torch.save(model.state_dict(),model_path)
        
    plt.figure()
    plt.plot(train_accs,label='train')
    plt.plot(test_accs,label='test')
    plt.legend()
    plt.savefig(exp_path)


