import torch
import torch.nn as nn
import numpy as np
import random, time
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
from ogb.graphproppred.mol_encoder import BondEncoder
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_adj
import copy
import torch.nn.functional as F
import math
from typing import Optional
from torch import Tensor
from torch_scatter import scatter
from torchmetrics.functional import pairwise_cosine_similarity
import os, pickle
import ot 
from typing import Any, Optional, Tuple, Union

import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.utilities.compute import _safe_matmul
from torchmetrics.functional.pairwise.helpers import _check_input, _reduce_distance_matrix
import argparse
from typing import Optional
from torch import Tensor
from sklearn.cluster import KMeans


EPS = 1e-15

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv1d):
            print(m)
            m.weight.data.normal_(0, 0.02)
            # m.bias.data.zero_()
    
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def save_result(result_csv, dataset, subject, acc, f1, class_f1, seed, epoch, spt_lr, file_name):
    df_col = [
        'dataset',
        'subject',
        'acc',
        'f1',
        'class_f1',
        'random_seed',
        'epoch',
        'spt_lr',
        'file_name'
    ]

    if not os.path.exists(result_csv):
        print(result_csv)
        df = pd.DataFrame([], columns= df_col)
        print(df)
        df.to_csv(result_csv, header=True, index=False)

    data = {
        'dataset' : dataset,
        'subject' : subject,
        'acc' : acc,
        'f1' : f1,
        'class_f1': class_f1,
        'random_seed' : seed,
        'epoch' : epoch,
        'spt_lr' : spt_lr,
        'file_name' : file_name
    }
    result_df = pd.DataFrame.from_records([data])
    result_df.to_csv(result_csv, mode='a', header=False, index=False)

def global_add_pool(x: Tensor, batch: Optional[Tensor],
                    size: Optional[int] = None) -> Tensor:
    if batch is None:
        return x.sum(dim=-2, keepdim=x.dim() == 2)
    size = int(batch.max().item() + 1) if size is None else size
    return scatter(x, batch, dim=-2, dim_size=size, reduce='add')

def global_mean_pool(x: Tensor, batch: Optional[Tensor],
                     size: Optional[int] = None) -> Tensor:
    if batch is None:
        return x.mean(dim=-2, keepdim=x.dim() == 2)
    size = int(batch.max().item() + 1) if size is None else size
    return scatter(x, batch, dim=-2, dim_size=size, reduce='mean')

def addNoise(data, noise_scale_rate=0.5):
    return data + torch.randn(size = data.shape, device=data.device)


def sigma(dists, kth=8):
    # Compute sigma and reshape
    try:
        # Get k-nearest neighbors for each node
        knns = np.partition(dists, kth, axis=-1)[:, kth::-1]
        sigma = knns.sum(axis=1).reshape((knns.shape[0], 1))/kth
    except ValueError:     # handling for graphs with num_nodes less than kth
        num_nodes = dists.shape[0]
        # this sigma value is irrelevant since not used for final compute_edge_list
        sigma = np.array([1]*num_nodes).reshape(num_nodes,1)
        
    return sigma + 1e-8 # adding epsilon to avoid zero value of sigma


def construct_adjacency(x, distance = 'euclidean', mu_val=0.1, adj_norm = True):
    if distance =='euclidean':
        dist = pairwise_euclidean_distance(x)
        std = dist.std()
        adj = torch.exp(-torch.square(dist/std))
    elif distance =='cosine':
        dist = 1 - pairwise_cosine_similarity(x, zero_diagonal=False)    
        std = dist.std()
        adj = torch.exp(-torch.square(dist/std))
    elif distance == 'gaussian_kernel':
        adj = gaussian_kernel(x,x, mu=mu_val)

    if adj_norm == True:
        adj = adj.unsqueeze(0)
        ind = torch.arange(adj.size(1), device=adj.device)
        adj[:, ind, ind] = 0
        d = torch.einsum('ijk->ij', adj)
        d = torch.sqrt(d)[:, None] + EPS
        adj = (adj / d) / d.transpose(1, 2)
        adj[:, ind, ind] = 1

    return adj

def construct_adjacency_norm(x, distance = 'euclidean'):
    if distance =='euclidean':
        dist = pairwise_euclidean_distance(x)
    if distance =='cosine':
        dist = 1 - pairwise_cosine_similarity(x, zero_diagonal=False)    
    std = dist.std()
    adj = torch.exp(-torch.square(dist/sigma(dist)**2))

    return adj


def _pairwise_euclidean_distance_update(
    x: Tensor, y: Optional[Tensor] = None, zero_diagonal: Optional[bool] = None
) -> Tensor:
    x, y, zero_diagonal = _check_input(x, y, zero_diagonal)
    # upcast to float64 to prevent precision issues
    _orig_dtype = x.dtype
    x = x.to(torch.float64)
    y = y.to(torch.float64)
    x_norm = (x * x).sum(dim=1, keepdim=True)
    y_norm = (y * y).sum(dim=1)
    distance = (x_norm + y_norm - 2 * x.mm(y.T)).to(_orig_dtype)
    if zero_diagonal:
        distance.fill_diagonal_(0)
    return distance.sqrt()


def pairwise_euclidean_distance(
    x: Tensor,
    y: Optional[Tensor] = None,
    reduction: Literal["mean", "sum", "none", None] = None,
    zero_diagonal: Optional[bool] = None,
) -> Tensor:
    distance = _pairwise_euclidean_distance_update(x, y, zero_diagonal)
    return _reduce_distance_matrix(distance, reduction)



def _pairwise_cosine_similarity_update(
    x: Tensor, y: Optional[Tensor] = None, zero_diagonal: Optional[bool] = None
) -> Tensor:
    x, y, zero_diagonal = _check_input(x, y, zero_diagonal)

    norm = torch.norm(x, p=2, dim=1)
    x = x / norm.unsqueeze(1)
    norm = torch.norm(y, p=2, dim=1)
    y = y / norm.unsqueeze(1)

    distance = _safe_matmul(x, y)
    if zero_diagonal:
        distance.fill_diagonal_(0)
    return distance

# modifying NaN bug
def pairwise_cosine_similarity(
    x: Tensor,
    y: Optional[Tensor] = None,
    reduction: Literal["mean", "sum", "none", None] = None,
    zero_diagonal: Optional[bool] = None,
) -> Tensor:
    distance = _pairwise_cosine_similarity_update(x, y, zero_diagonal)
    return _reduce_distance_matrix(distance, reduction)



def loss_plot(hist, path='Train_hist.png', model_name=''):
    x = range(len(hist['lp_loss']))

    y1 = hist['lp_loss']
    y2 = hist['entropy_loss']
    y3 = hist['ssl_loss']
    y4 = hist['total_loss']

    plt.plot(x, y1, label='lp_loss')
    plt.plot(x, y2, label='entropy_loss')
    plt.plot(x, y3, label='ssl_loss')
    plt.plot(x, y4, label='total_loss')


    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(path, model_name + '_loss.png')

    plt.savefig(path)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di



def dense_to_sparse(adj, device = 'cuda'):
    # from pyg
    assert adj.dim() >= 2 and adj.dim() <= 3
    assert adj.size(-1) == adj.size(-2)

    x = torch.arange(0, adj.size(0))
    idx_1 = x.repeat_interleave(adj.size(0))
    idx_2 = x.repeat(adj.size(0))    
    idxs = idx_1, idx_2
    edge_attr = adj[idxs]

    if len(idxs) == 3:
        batch = idxs[0] * adj.size(-1)
        idxs = (batch + idxs[1], batch + idxs[2])

    return torch.stack(idxs, dim=0).to(device), edge_attr

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def minmax_by_channel(x):
    num_channel = x.shape[0]
    x_scaled = np.zeros([x.shape[0], x.shape[1], x.shape[2]])
    for i in range(num_channel):
        x_min = x[i, :,:].min()
        x_max = x[i, :,:].max()
        x_scaled[i,:,:] = (x[i,:,:] - x_min) / (x_max - x_min)

    return x_scaled

def std_by_channel(x, per_epoch=False):
    num_channel = x.shape[0]

    if per_epoch:
        x_scaled = np.zeros([x.shape[0], x.shape[1]])
        for i in range(num_channel):
            mean = x[i, :].mean()
            std = x[i, :].std()
            x_scaled[i,:] = (x[i,:] - mean) / std

    else: # per subject 
        x_scaled = np.zeros([x.shape[0], x.shape[1], x.shape[2]])
        for i in range(num_channel):
            mean = x[i, :,:].mean()
            std = x[i, :,:].std()
            x_scaled[i,:,:] = (x[i,:,:] - mean) / std        

    return x_scaled


def linearKernel(x , y):
    x_flat=torch.flatten(x,start_dim=1)
    y_flat=torch.flatten(y,start_dim=1)
    return x_flat@y_flat.T

def gaussian_kernel(x, y, mu=0.1):
    x = torch.flatten(x, start_dim = 1)
    y = torch.flatten(y, start_dim = 1)
    distance = ot.dist(x, y)
    return torch.exp(-distance / mu)


def _sim_matrix(a, b, eps=EPS):
    """
    computing cosine similarity  for input vectors 
    
    Args:
        a: input vector
        b: input vector
        eps: numerical stability 
    return: 
         cos(a, b)
    """
    # get norm(a) & norm(b)
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]

    # get a / norm(a) & b / norm(b)
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))

    # calculate cosine similarity
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

def sim_matrix(data, edges, metric ='cosine', device='cuda', eps=EPS):
    '''
    compute cosine similairty matrix for given edges 

    Args:
        data [Tensor]: input data
        edges [Tensor]: input edges (before .t().contingous())
        mtric [str]: distance metric [cosine / linear_kernel / gaussian_kernel]
        device: cuda or cpu
        eps: numerical stability

    Return:
        sim_weights_batch [Tensor] : cosine similairty matrix of all # seq_length (#epochs)

        e.g., 
            if input data is (1, 700, 6, 3000), 
            (a) Compute similairty (6,3000) 
            (b) Select values correspond to the given edge among them.
            repeat (a) & (b) for all 700 seq_length, 
            
    '''

    sim_weights_batch = torch.tensor(()).to(device)

    for i in range(len(data)):
        
        # (a)
        if metric =='cosine':
            output = _sim_matrix(data[i], data[i], eps) 
        elif metric =='gaussian_kernel':
            output = gaussian_kernel(data[i], data[i])
        elif metric == 'linear_kernel':
            output = linearKernel(data[i], data[i])

        weights = torch.tensor(()).to(device) # (b)
        for edge in edges:
            out = output[edge[0], edge[1]].unsqueeze(0)
            weights = torch.cat((weights, out), dim=0)

        weights = weights.unsqueeze(0)
        # [# batch (epochs), # edges ] , e.g., [766, 18]
        sim_weights_batch  = torch.cat((sim_weights_batch, weights), dim=0) 
    
    # Assign 'eps' at negative cosine similarity value.
    sim_weights_batch = torch.max(eps*torch.ones_like(sim_weights_batch), sim_weights_batch)
    return sim_weights_batch


def remove(tensor, idx , device = 'cuda'):
    '''
    Remove tensor[idx] item from input tensor 

    Args:
        tensor (torch.Tensor): input tensor
        idx (int): index of item to be removed
    
    Returns:
        torch.Tensor: output tensor
    '''
    mask = torch.ones(tensor.numel(), dtype=torch.bool).to(device)
    mask = mask.reshape(-1, 2)
    mask[idx] = False
    return tensor[mask].reshape(-1,2)

def random_drop_edge(data_shape, edges_matrix, ratio=0.1, device = 'cuda'):
    
    '''
    Generate Randomly dropped edge 

    Args:
        data_shape (int): # nodes (for isruc, 6)
        edges_matrix (torch.Tensor): edge matrix
        ratio (float, optional): ratio of dropping edge
    
    Returns:
        torch.Tensor: randomly dropped edges.  
    '''
    # Create edge mask 
    edge_mask = torch.triu_indices(data_shape, data_shape, offset=1).t()
    num_dropouts = np.int32(np.ceil(len(edge_mask) * ratio)) 
    rand_idx = np.random.choice(range(0, data_shape -1), num_dropouts, replace=False)

    for idx in rand_idx: 
        # remove edges symmetrically 
        masked_edge = remove(edges_matrix, idx, device= device) 
        masked_edge = remove(masked_edge, idx, device= device) 
    masked_edge = masked_edge.type(torch.LongTensor).to(device)
    return masked_edge


def batched_random_drop_edge(num_batch, num_channels, edge_index, ratio=0.1, device='cuda'):
    batched_drop_edge = torch.tensor(()).to(device)

    for _ in range(num_batch):
        edge = edge_index.clone()
        masked_edge = random_drop_edge(num_channels, edge, ratio, device=device)
        batched_drop_edge = torch.cat((batched_drop_edge, masked_edge.unsqueeze(0)), dim=0)
    
    batched_drop_edge = batched_drop_edge.transpose(1,2)
    #print("batched_drop_edge", batched_drop_edge.shape)
    
    return batched_drop_edge.type(torch.LongTensor).to(device)


def batched_sim_matrix(data, edges, metric='cosine', eps=EPS, device='cuda'):
    '''
    Args: 
        data (torch.Tensor): input data for calculate cosine similarity (1, #sequence, # nodes, fft dim)
        edges (torch.Tensor): edges matrix (1, #sequence, 2, # edge)
    
    Returns:
        torch.Tensor: batched sim matrix (#sequence, #edges )
    '''
    sim_weights_batch = torch.tensor(()).to(device)

    for i in range(len(data)):
        
        if metric =='cosine':
            output = _sim_matrix(data[i], data[i], eps) 

        elif metric =='gaussian_kernel':
            output = gaussian_kernel(data[i], data[i])
        elif metric == 'linear_kernel':
            output = linearKernel(data[i], data[i])
        elif metric == 'gaussian_v2':
            output = gaussian_kernel_v2(data[i], data[i])

        
        
        weights = torch.tensor(()).to(device) # (b)
        for edge in edges[i]:
            out = output[edge[0], edge[1]].unsqueeze(0)
            weights = torch.cat((weights, out), dim=0)

        weights = weights.unsqueeze(0)
        # [# batch (epochs), # edges ] , e.g., [766, 18]
        sim_weights_batch  = torch.cat((sim_weights_batch, weights), dim=0) 
    
    # Assign 'eps' at negative cosine similarity value.
    sim_weights_batch = torch.max(eps*torch.ones_like(sim_weights_batch), sim_weights_batch)
    return sim_weights_batch


def fft(x):
    
    signal = x.reshape((x.shape[0], x.shape[1]))
    signal = np.fft.rfft(signal,  x.shape[1], axis=1)
    signal = np.abs(signal)
    signal[signal == 0.0] = 1e-8 
    signal = np.log(signal)
    return signal

