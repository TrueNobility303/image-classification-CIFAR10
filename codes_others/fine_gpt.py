import numpy as np
from torch.nn.modules.activation import ReLU
from torch.nn.modules.linear import Linear
from torch.optim import optimizer
from torch.utils.data import dataloader
import torchvision
import torch
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from net.gpt import GPT,GPTConfig
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

BATCH = 16
transform = transforms.Compose( [transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='./datasets', train=True, download=True)
testset = torchvision.datasets.CIFAR10(root='./datasets', train=False, download=True)
trainset = PartialDataset(trainset,1000)
testset = PartialDataset(testset,100)

pluck_rgb = lambda x: torch.from_numpy(np.array(x)).view(32*32, 3)[torch.randperm(32*32)[:5], :]
px = torch.cat([pluck_rgb(x) for x, y in trainset], dim=0).float()

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

class ImageDataset(Dataset):
    def __init__(self, pt_dataset, clusters, perm=None):
        self.pt_dataset = pt_dataset
        self.clusters = clusters
        self.perm = torch.arange(32*32) if perm is None else perm
        
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

train_dataset = ImageDataset(trainset, ClUSTER_DICT)
test_dataset = ImageDataset(testset, ClUSTER_DICT)

#配置GPT模型，并且定义模型
mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                  embd_pdrop=0.0, resid_pdrop=0.0, attn_pdrop=0.0,
                  n_layer=6, n_head=4, n_embd=128)
model = GPT(mconf).to(device)
model.load_state_dict(torch.load('pth/gpt.pth'))

#参数，token,pos,trasnformer,并修改head结构
for p in model.tok_emb.parameters():
    p.requires_grad = False
model.pos_emb.requires_grad = False
for p in model.blocks.parameters():
    p.requires_grad = False
model.head = nn.Sequential(
    nn.Linear(mconf.n_embd, 1024),
    nn.ReLU(),
    nn.Linear(1024,4096),
    nn.ReLU(),
    nn.Linear(4096,10))
for p in model.head.parameters():
    p.requires_grad = True
model.to(device)
#print(model)

#recap transformer结构,详见gpt.py

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH, shuffle=False)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
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

        #取均值再取平均，等价于全局均值池化GAP，得到logits
        logits,_ = model(x)
        logits = logits.mean(1)
        logits = torch.sigmoid(logits)
        
        loss = critirion(logits,label) 
        loss.backward()
        optimizer.step()

        tot_loss += loss.item()
        tot_num += label.shape[0]

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
        logits = logits.mean(1)
        logits = torch.sigmoid(logits)

        pred = logits.argmax(-1)
        tot_correct += torch.sum(pred==label).item() 

        tot_num += label.shape[0]

    return tot_correct / tot_num

if __name__ == '__main__':
    train_accs = []
    test_accs = []
    
    exp_path = 'exp/fine_gpt.png'

    for epoch in range(200):
        loss = train(trainloader)
        acc = test(testloader)
        train_acc = test(trainloader)
        print('epoch',epoch,'loss',loss,'train',train_acc,'test',acc)
        test_accs.append(acc)
        train_accs.append(train_acc)

    plt.figure()
    plt.plot(train_accs)
    plt.plot(test_accs)
    plt.legend(['train','test'])
    plt.savefig(exp_path)




