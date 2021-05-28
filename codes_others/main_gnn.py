from torchvision.datasets import MNIST
import torchvision.transforms as T
from torch_geometric.transforms import ToSLIC
import torchvision
import torch

import torch_geometric
from torch_geometric.utils import from_networkx 
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear

import tqdm 
import numpy as np
import matplotlib.pyplot as plt 
from torch_geometric.data import DataLoader
from torch_cluster import knn_graph

device = torch.device('cuda')
BATCH = 512
transform = T.Compose( [T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5,0.5)), ToSLIC(n_segments=128)])
trainset = torchvision.datasets.CIFAR10(root='./datasets', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=BATCH, shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./datasets', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=BATCH, shuffle=False)

torch.manual_seed(42)

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_node_features=3, num_classes=10):
        super(GCN, self).__init__()
       
        self.conv1 = GATConv(num_node_features, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)
        self.conv4 = GATConv(hidden_channels, hidden_channels)
        self.conv5 = GATConv(hidden_channels, hidden_channels)

        self.lin = nn.Sequential(   
            Linear(hidden_channels,1024),
            nn.ReLU(),
            Linear(1024,num_classes) 
        )

    def forward(self, x, edge_index,batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        #x = self.conv4(x, edge_index)
        #x = x.relu()
        #x = self.conv5(x, edge_index)
        #print(x.shape)
        
        x = global_mean_pool(x,batch)  
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

model = GCN(128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = torch.nn.CrossEntropyLoss()

def train(dataloader):
    model.train()
    tot_loss = 0 
    num = 0
    for i,data in tqdm.tqdm(enumerate(trainloader)):   
        #构建KNN图，K为超参数
        graph,y = data
        graph = graph.to(device)
        y = y.to(device)
        num += len(y)
        edge_index = knn_graph(graph.pos, k=6)
        logits = model(graph.x, edge_index,graph.batch)
        pred = torch.argmax(logits,1)
        loss = criterion(logits,y)
        tot_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return tot_loss / num 

def test(dataloader):
    model.eval()
    tot_correct = 0 
    num = 0
    for i,data in tqdm.tqdm(enumerate(trainloader)):   
        #构建KNN图，K为超参数
        graph,y = data
        graph = graph.to(device)
        y = y.to(device)
        num += len(y)
        edge_index = knn_graph(graph.pos, k=6)
        logits = model(graph.x, edge_index,graph.batch)
        pred = torch.argmax(logits,1)
        tot_correct += torch.sum(pred==y).item()

    return tot_correct / num 

if __name__ == '__main__':
    model_path = 'pth/gnn_cifar.pth'
    exp_path = 'exp/gnn_cifar.png'
    train_accs = []
    test_accs = []
    for e in range(100):
        trainacc = test(trainloader)
        testacc = test(testloader)
        loss = train(trainloader)
        train_accs.append(trainacc)
        test_accs.append(testacc)
        
        print('epoch',e,'loss',loss, 'train',trainacc,'test',testacc)
        torch.save(model.state_dict(),model_path)

    plt.figure()
    plt.plot(train_accs)
    plt.plot(test_accs)
    plt.legend(['train','test'])
    plt.savefig(exp_path)


