import numpy as np
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
#print('finsh kmeans')

n_samples = 16
ncol = 8
nrow = n_samples // ncol + 1
plt.figure(figsize=(20, 10))
for i in range(n_samples):
    x, y = trainset[np.random.randint(0, len(trainset))]
    xpt = torch.from_numpy(np.array(x)).float().view(32*32, 3)
    ix = ((xpt[:, None, :] - ClUSTER_DICT[None, :, :])**2).sum(-1).argmin(1) # cluster assignments for each pixel
    
    plt.subplot(nrow, ncol, i+1)
    plt.imshow(ClUSTER_DICT[ix].view(32, 32, 3).numpy().astype(np.uint8))
    plt.axis('off')
plt.savefig('dump/kmeans_alg.png')

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
        #Bx1023x512
        return a[:-1], a[1:] 

train_dataset = ImageDataset(trainset, ClUSTER_DICT)
test_dataset = ImageDataset(testset, ClUSTER_DICT)

#配置GPT模型，并且定义模型
mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                  embd_pdrop=0.0, resid_pdrop=0.0, attn_pdrop=0.0,
                  n_layer=6, n_head=4, n_embd=128)
model = GPT(mconf).to(device)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH, shuffle=False)
model_path = 'pth/gpt.pth'
exp_path = 'exp/gpt.png'

#定义学习器及相关超参数
optimizier = model.configure_optimizers(lr=3e-3,weight_decay=5e-4)

#训练时选取最好的k个样本
def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] 
        logits, _ = model(x_cond)
        logits = logits[:, -1, :] / temperature
        #选取topk或者argmax
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        probs = F.softmax(logits, dim=-1)
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)

        x = torch.cat((x, ix), dim=1)

    return x

def train(dataloader):
    model.train()
    tot_loss = 0
    tot_num = 0
    tot_correct = 0
    for i,data in tqdm.tqdm(enumerate(dataloader)):
        optimizier.zero_grad()
        x,y = data
        x = x.to(device)
        y = y.to(device)

        logits,loss = model(x,y)
        pred = logits.argmax(-1)
        tot_correct += torch.sum(pred==y).item() 
        loss.backward()
        optimizier.step()

        tot_loss += loss.item()
        tot_num += x.shape[0] * x.shape[1]

    return tot_loss / tot_num, tot_correct / tot_num

def test(train_dataset, savepath):
    counts = torch.ones(NCLUSTER) 
    rp = torch.randperm(len(train_dataset))
    nest = 5000 
    for i in range(nest):
        a, _ = train_dataset[int(rp[i])]
        t = a[0].item() 
        counts[t] += 1
        prob = counts/counts.sum()

    #估计prob并作为序列的开头     
    model.eval()
    n_samples = 32
    start_pixel = np.random.choice(np.arange(ClUSTER_DICT.size(0)), size=(n_samples, 1), replace=True, p=prob)
    start_pixel = torch.from_numpy(start_pixel).to(device)
    pixels = sample(model, start_pixel, 32*32-1, temperature=1.0, sample=True, top_k=100)
    iperm = torch.argsort(train_dataset.perm)

    ncol = 8
    nrow = n_samples // ncol
    plt.figure(figsize=(16, 8))
    for i in range(n_samples):
        pxi = pixels[i][iperm] 
        plt.subplot(nrow, ncol, i+1)
        plt.imshow(ClUSTER_DICT[pxi].view(32, 32, 3).numpy().astype(np.uint8))
        plt.axis('off')
    plt.savefig(savepath)

if __name__ == '__main__':
    train_accs = []
    test_accs = []
    for epoch in range(200):
        loss,trainacc = train(trainloader)
        train_accs.append(trainacc)

        path = 'dump_gpt/' + str(epoch) + '.png' 
        test(test_dataset,path)

        if epoch % 1 == 0:
            print('epoch',epoch,'loss',loss,'acc',trainacc)
            torch.save(model.state_dict(),model_path)
    plt.figure()
    plt.plot(train_accs)
    plt.plot(test_accs)
    plt.legend(['train','test'])
    plt.savefig(exp_path)




