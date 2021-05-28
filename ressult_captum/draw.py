import numpy as np
import matplotlib.pyplot as plt
from numpy.core.einsumfunc import _parse_possible_contraction

#lrs = [1e-4, 5e-4, 1e-3, 2e-3, 3e-3]

vgg_bn_grad_lst = np.load('landscape/vgg_bn_grad_lst.npy',allow_pickle=True)
vgg_bn_loss_lst = np.load('landscape/vgg_bn_loss_lst.npy',allow_pickle=True)
vgg_bn_train_lst = np.load('landscape/vgg_bn_train_lst.npy',allow_pickle=True)
vgg_bn_val_lst = np.load('landscape/vgg_bn_val_lst.npy',allow_pickle=True)
vgg_bn_wei_lst = np.load('landscape/vgg_bn_wei_lst.npy',allow_pickle=True)

vgg_grad_lst = np.load('landscape/vgg_grad_lst.npy',allow_pickle=True)
vgg_loss_lst = np.load('landscape/vgg_loss_lst.npy',allow_pickle=True)
vgg_train_lst = np.load('landscape/vgg_train_lst.npy',allow_pickle=True)
vgg_val_lst = np.load('landscape/vgg_val_lst.npy',allow_pickle=True)
vgg_wei_lst = np.load('landscape/vgg_wei_lst.npy',allow_pickle=True)

def draw_loss_landscape(savepath):
    plt.style.use('ggplot')
    plt.clf()
    interval = 10

    #画图时修改interval
      
    max_loss = np.max(vgg_loss_lst[:3],0)[::interval]
    min_loss = np.min(vgg_loss_lst[:3],0)[::interval]
    
    x = np.arange(0,len(max_loss),1)
    plt.plot(x, max_loss, c='deeppink')
    plt.plot(x, min_loss, c='deeppink')
    plt.fill_between(x, max_loss, min_loss ,facecolor='lightpink', label='without BN')

    max_bn_loss = np.max(vgg_bn_loss_lst[:3],0)[::interval]
    min_bn_loss = np.min(vgg_bn_loss_lst[:3],0)[::interval]
    plt.plot(x, max_bn_loss, c='mediumseagreen')
    plt.plot(x, min_bn_loss, c='mediumseagreen')
    plt.fill_between(x, max_bn_loss, min_bn_loss, facecolor="lightgreen", label = 'with BN')
    plt.legend()
    plt.savefig(savepath)

def draw_predgrad(savepath):
    plt.style.use('ggplot')
    plt.clf()
    interval = 10
    n_lr,n_iter,_,_ = vgg_bn_grad_lst.shape
    grad_diff = np.zeros((n_lr,n_iter-1))
    for i in range(n_lr):
        for j in range(n_iter-1):
            grad_diff[i,j] = np.linalg.norm(vgg_grad_lst[i,j+1,:] - vgg_grad_lst[i,j,:])
    
    max_grad_diff = np.max(grad_diff[:3],0)[::interval]
    min_grad_diff = np.min(grad_diff[:3],0)[::interval]
    x = np.arange(0,len(max_grad_diff),1)
    begin = 1
    x = x[begin:]
    max_grad_diff = max_grad_diff[begin:]
    min_grad_diff = min_grad_diff[begin:]
    plt.plot(x, max_grad_diff, c='mediumseagreen')
    plt.plot(x, min_grad_diff, c='mediumseagreen')
    plt.fill_between(x,max_grad_diff,min_grad_diff,facecolor='lightgreen', label='with BN')

    grad_diff = np.zeros((n_lr,n_iter-1))
    for i in range(n_lr):
        for j in range(n_iter-1):
            grad_diff[i,j] = np.linalg.norm(vgg_bn_grad_lst[i,j+1,:] - vgg_bn_grad_lst[i,j,:])
    max_grad_diff = np.max(grad_diff[:3],0)[::interval]
    min_grad_diff = np.min(grad_diff[:3],0)[::interval]
    x = np.arange(0,len(max_grad_diff),1)
    x = x[begin:]
    max_grad_diff = max_grad_diff[begin:]
    min_grad_diff = min_grad_diff[begin:]
    plt.plot(x, max_grad_diff, c='deeppink')
    plt.plot(x, min_grad_diff, c='deeppink')
    plt.fill_between(x,max_grad_diff,min_grad_diff,facecolor='lightpink', label='without BN')

    plt.legend()
    plt.savefig(savepath)

def draw_beta_smooth(savepath):

    plt.style.use('ggplot')
    plt.clf()
    interval = 10
    n_lr,n_iter,_,_ = vgg_bn_grad_lst.shape
    grad_diff = np.zeros((n_lr,n_iter-1))
    for i in range(n_lr):
        for j in range(n_iter-1):
            grad_diff[i,j] = np.linalg.norm(vgg_grad_lst[i,j+1,:] - vgg_grad_lst[i,j,:])
    
    max_grad_diff = np.max(grad_diff[:],0)[::interval]
    min_grad_diff = np.min(grad_diff[:],0)[::interval]
    x = np.arange(0,len(max_grad_diff),1)
    begin = 1
    x = x[begin:]
    max_grad_diff = max_grad_diff[begin:]
    min_grad_diff = min_grad_diff[begin:]
    plt.plot(x, max_grad_diff, c='mediumseagreen', label='with BN')
    #plt.plot(x, min_grad_diff, c='mediumseagreen')
    #plt.fill_between(x,max_grad_diff,min_grad_diff,facecolor='lightgreen', label='without BN')

    grad_diff = np.zeros((n_lr,n_iter-1))
    for i in range(n_lr):
        for j in range(n_iter-1):
            grad_diff[i,j] = np.linalg.norm(vgg_bn_grad_lst[i,j+1,:] - vgg_bn_grad_lst[i,j,:])
    max_grad_diff = np.max(grad_diff[:],0)[::interval]
    min_grad_diff = np.min(grad_diff[:],0)[::interval]
    x = np.arange(0,len(max_grad_diff),1)
    x = x[begin:]
    max_grad_diff = max_grad_diff[begin:]
    min_grad_diff = min_grad_diff[begin:]
    plt.plot(x, max_grad_diff, c='deeppink', label='without BN')
    #plt.plot(x, min_grad_diff, c='deeppink')
    #plt.fill_between(x,max_grad_diff,min_grad_diff,facecolor='lightpink', label='with BN')

    plt.legend()
    plt.savefig(savepath)

if __name__ == '__main__':
    draw_loss_landscape('landscape/loss_landscape.png')
    draw_predgrad('landscape/pred_grad.png')
    draw_beta_smooth('landscape/beta_smooth.png')