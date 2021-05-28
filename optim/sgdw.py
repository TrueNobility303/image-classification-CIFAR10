import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer, required
from typing import List, Optional
from torch import Tensor

#SGD from pytorch 
def sgdw(params: List[Tensor],
        d_p_list: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        momentum: float,
        weight_decay: float,
        lr: float):

    for i, param in enumerate(params):
        d_p = d_p_list[i]

        #使用W方式替换权重衰减部分
        param.mul_(1 - lr * weight_decay)

        #动量
        if momentum != 0:
            buf = momentum_buffer_list[i]
            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            buf.mul_(momentum).add_(d_p, alpha=1)
            d_p = buf

        param.add_(d_p, alpha=-lr)

class mySGDW(Optimizer):
    def __init__(self, params, lr=required, momentum=0.9,weight_decay=0):

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    @torch.no_grad()
    def step(self):
        loss = None
        
        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            momentum = group['momentum']
            lr = group['lr']
            weight_decay = group['weight_decay']

            #将所有的参数p，对应的梯度存入lst中，在state[p]中记录momemtum
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            #将dp存入dp_lst
            sgdw(params_with_grad,
                  d_p_list,
                  momentum_buffer_list,
                  momentum=momentum,
                  weight_decay=weight_decay,
                  lr=lr)

            #在momentum buffer 中记录p的上一步V的值
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer
                
        return loss
