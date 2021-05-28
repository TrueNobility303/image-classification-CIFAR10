import numpy as np
import torchvision
import torch
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from config import device
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.nn import functional as F

#截取部分数据集
class PartialDataset(Dataset):
    def __init__(self, dataset, n_items):
        self.dataset = dataset
        self.n_items = n_items

    def __getitem__(self,index):
        return self.dataset.__getitem__(index)

    def __len__(self):
        return min(self.n_items, len(self.dataset))

#设置随机种子，使得代码可复现
def set_random_seeds(seed_value=0):
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False