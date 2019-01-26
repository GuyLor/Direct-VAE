import torch
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import math
import numpy as np
import os
import time

is_cuda = torch.cuda.is_available()

def denorm(x): #from range [-1,1] to [0,1]
    out = (x + 1) / 2
    return out.clamp(0, 1)

def norm(x): #from range [0,1] to [-1,1]
    out = 2*x - 1
    return out.clamp(-1, 1)

def duplicate_along(X,D):
    y = X.unsqueeze(2).repeat(1, 1, D)
    z = y.transpose(1,2).contiguous().view(X.size(0),-1)
    return z

def to_var(x):
    if is_cuda:
        x = x.cuda()
    return x

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print ('%s function took %0.3f ms' % (f.func_name, (time2-time1)*1000.0))
        return ret
    return wrap
    
def print_results(epoch_results,epoch,num_epochs):
    print (" ------ Epoch[{}/{}] ------ ".format(epoch+1,num_epochs))
    if len(epoch_results)==3:
        print ("Train NLL: {:.2f}".format(epoch_results[0]))
        print ("Valid NLL: {:.2f}".format(epoch_results[1]))
        print ("Test  NLL: {:.2f}".format(epoch_results[2]))
    else:
        print ("Train NLL: {:.2f}".format(epoch_results[0]))
        print ("Test  NLL: {:.2f}".format(epoch_results[1]))
    
def kl_multinomial(logits_z):
    M = logits_z.size(-1)
    q_z = F.softmax(logits_z,dim =1)
    log_q_z = torch.log(q_z+1e-20)
    kl_tmp = q_z * (log_q_z - math.log(1.0/M))
    return kl_tmp.sum()

def kl_gaussian(mu,log_var):
    return torch.sum(0.5 * (mu**2 + torch.exp(log_var) - log_var -1))

class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)
    def __iter__(self):
        return iter(self._modules.values())
    def __len__(self):
        return len(self._modules)
