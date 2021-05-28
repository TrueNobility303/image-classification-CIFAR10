import tqdm
import time
import pickle
import multiprocessing
import numpy as np
import scipy as sp
from skimage.segmentation import slic, mark_boundaries
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from config import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear

from torchvision.datasets import CIFAR10
import torchvision
import torchvision.transforms as transforms

import torch_geometric
from torch_geometric.utils import from_networkx 
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool

# from https://github.com/phcavelar/cifar-superpixel/blob/master/util.py

def get_graph_from_image(PIL_image,desired_nodes=512):
    image = np.asarray(PIL_image)
    w,h,c = image.shape
    if c == 1:
        new_img = np.zeros((w,h,3))
        new_img[:,:,0] = image[:,:,0]
        new_img[:,:,1] = image[:,:,0]
        new_img[:,:,2] = image[:,:,0]
        image = new_img
    segments = slic(image, n_segments=desired_nodes, slic_zero = True, start_label=0)
    asegments = np.array(segments)

    num_nodes = np.max(asegments)
    nodes = {
        node: {
            "rgb_list": [],
            "pos_list": []
        } for node in range(num_nodes+1)
    }

    height = image.shape[0]
    width = image.shape[1]
    for y in range(height):
        for x in range(width):
            node = asegments[y,x]
            rgb = image[y,x,:]
            pos = np.array([float(x)/width,float(y)/height])
            nodes[node]["rgb_list"].append(rgb)
            nodes[node]["pos_list"].append(pos)
    
    G = nx.Graph()
    
    for node in nodes:
        nodes[node]["rgb_list"] = np.stack(nodes[node]["rgb_list"])
        nodes[node]["pos_list"] = np.stack(nodes[node]["pos_list"])
        rgb_mean = np.mean(nodes[node]["rgb_list"], axis=0)
        #rgb_std = np.std(nodes[node]["rgb_list"], axis=0)
        #rgb_gra Posm = np.matmul( nodes[node]["rgb_list"].T, nodes[node]["rgb_list"] ) / nodes[node]["rgb_list"].shape[0]
        pos_mean = np.mean(nodes[node]["pos_list"], axis=0)
        #pos_std = np.std(nodes[node]["pos_list"], axis=0)
        #pos_gram = np.matmul( nodes[node]["pos_list"].T, nodes[node]["pos_list"] ) / nodes[node]["pos_list"].shape[0]
        
        features = np.concatenate(
          [
            np.reshape(rgb_mean, -1),
            #np.reshape(rgb_std, -1),
            #np.reshape(rgb_gram, -1),
            np.reshape(pos_mean, -1),
            #np.reshape(pos_std, -1),
            #np.reshape(pos_gram, -1)
          ]
        )
        G.add_node(node, features = list(features))
    
    # From https://stackoverflow.com/questions/26237580/skimage-slic-getting-neighbouring-segments
    segments_ids = np.unique(segments)
    centers = np.array([np.mean(np.nonzero(segments==i),axis=1) for i in segments_ids])

    vs_right = np.vstack([segments[:,:-1].ravel(), segments[:,1:].ravel()])
    vs_below = np.vstack([segments[:-1,:].ravel(), segments[1:,:].ravel()])
    bneighbors = np.unique(np.hstack([vs_right, vs_below]), axis=1)

    #添加边和自环
    for i in range(bneighbors.shape[1]):
        if bneighbors[0,i] != bneighbors[1,i]:
            G.add_edge(bneighbors[0,i],bneighbors[1,i])
    for node in nodes:
        G.add_edge(node,node)
    G = from_networkx(G)
    return G

def plot_graph_from_image(PIL_image,desired_nodes=75,save_in=None):
    image = np.asarray(PIL_image)
    segments = slic(image, n_segments=desired_nodes, slic_zero = True)
    fig = plt.figure("Superpixels")
    print(image)
    ax = fig.add_subplot(1, 1, 1)
    #ax.imshow(mark_boundaries(image, segments), cmap="gray")
    ax.imshow(image)#, cmap="gray")
    plt.axis("off")

    asegments = np.array(segments)

    # From https://stackoverflow.com/questions/26237580/skimage-slic-getting-neighbouring-segments

    segments_ids = np.unique(segments)
    centers = np.array([np.mean(np.nonzero(segments==i),axis=1) for i in segments_ids])
    vs_right = np.vstack([segments[:,:-1].ravel(), segments[:,1:].ravel()])
    vs_below = np.vstack([segments[:-1,:].ravel(), segments[1:,:].ravel()])
    bneighbors = np.unique(np.hstack([vs_right, vs_below]), axis=1)

    plt.scatter(centers[:,1],centers[:,0], c='r')

    for i in range(bneighbors.shape[1]):
        y0,x0 = centers[bneighbors[0,i]]
        y1,x1 = centers[bneighbors[1,i]]
        l = Line2D([x0,x1],[y0,y1], c="r", alpha=0.5)
        ax.add_line(l)

    if save_in is None:
        plt.show()
    else:
        plt.savefig(save_in,bbox_inches="tight")
    plt.close()


BATCH  = 1
transform = transforms.Compose( [transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root='./datasets', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH, shuffle=True)
testset = torchvision.datasets.MNIST(root='./datasets', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH, shuffle=False)

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_node_features=5, num_classes=10):
        super(GCN, self).__init__()
        torch.manual_seed(42)
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

model = GCN(128)
optimizer = torch.optim.Adam(model.parameters(), lr=8e-5)
criterion = torch.nn.CrossEntropyLoss()

def train(dataloader):
    model.train()
    tot_loss = 0 
    num = 0
    for i,data in tqdm.tqdm(enumerate(trainloader)):   
        if i>1000:
            break 
        #仅支持BATCH为1，或者随机梯度下降法,TO DO: 支持全批量梯度下降
        x,y = data 
        x = x[0]
        y = y[0]
        num += 1
        x = x.permute(1,2,0)
        #plot_graph_from_image(x,desired_nodes=75,save_in='dump/superpixel.png')
        G = get_graph_from_image(x)
        batch = torch.zeros(1).long()
    
        logits = model(G.features.float(),G.edge_index,batch)
        pred = torch.argmax(logits,1)
        loss = criterion(logits,y.unsqueeze(0))
        tot_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return tot_loss / num 

def test(dataloader):
    model.eval()
    num = 0
    correct = 0
    for i,data in enumerate(trainloader):   
        if i>100:
            break 
        x,y = data 
        x = x[0]
        y = y[0]
        x = x.permute(1,2,0)
        #plot_graph_from_image(x,desired_nodes=75,save_in='dump/superpixel.png')
        G = get_graph_from_image(x)
        batch = torch.zeros(1).long()
    
        logits = model(G.features.float(),G.edge_index,batch)
        pred = torch.argmax(logits,1)
        num += 1
        #print(pred,y)
        correct += (pred[0]==y).item()
        loss = criterion(logits,y.unsqueeze(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    acc = correct / num
    return acc 

if __name__ == '__main__':
    for e in range(200):
        trainacc = test(trainloader)
        testacc = test(testloader)
        loss = train(trainloader)
        print('epoch',e,'loss',loss, 'train',trainacc,'test',testacc)