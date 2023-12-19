import torch
import numpy as np
import torch
import torch.nn as nn
import numpy as np

'''
code reference from : https://github.com/facebookresearch/barlowtwins/tree/main
'''

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class Barlow_loss(nn.Module):
    def __init__(self, out_dim = 128):
        super(Barlow_loss, self).__init__()
        #self.batchsize = batchsize
        self.bn = nn.BatchNorm1d(out_dim, affine=False) # normalize along batch dim
        
        self.fc = nn.Sequential(nn.Linear(128, 128), 
                                nn.BatchNorm1d(128),
                                nn.LeakyReLU(0.2))

    def forward(self, z1, z2):
        # z1(batch, z2)
        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2) 

        # sum the cross-correlation matrix between all gpus
        c.div_(z1.shape[0]) 

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        
        return on_diag, off_diag
