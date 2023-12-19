import numpy as np
from numpy import array, exp
from torch.nn.parameter import Parameter
import torch

class Gumbel(torch.nn.Module):
    def __init__(self, device):
        self.device = device
        super(Gumbel, self).__init__()

    def gumbel_sample(self, shape, eps=1e-20):
        u = torch.rand(shape)
        gumbel = - np.log(- np.log(u + eps) + eps)
        gumbel = gumbel.to(self.device)
        return gumbel

    def gumbel_softmax_sample(self, logits, temperature): 
        y = logits + self.gumbel_sample(logits.size())
        return torch.nn.functional.softmax( y / temperature, dim = 1)

    def gumbel_softmax(self, logits, temperature, hard=False):
        y = self.gumbel_softmax_sample(logits, temperature)
        if hard:
            y_hard = torch.max(y.data, 1)[1]
            y = y_hard
        return y

    def generate_adj(self, adj):
        adj = Parameter(adj)
        ones = torch.ones(adj.size(0), adj.size(0))
        ones = ones.to(self.device)
        adj_minus = abs(adj - ones)
        gen_matrix = torch.stack((adj, adj_minus),dim = 2)

        # gen_mat = Parameter(gen_matrix)
        return gen_matrix

    def sample(self, adj, temperature, hard=False):
        i,j = torch.triu_indices(adj.size(0), adj.size(0), offset=1)
        prob_ex = adj[i,j]
        prob_unex = abs(1-prob_ex)
        gumbel_input = torch.stack((prob_ex, prob_unex), dim=1)
        gumbel_output = self.gumbel_softmax(gumbel_input, temperature, hard)

        if hard:
            out = torch.zeros(adj.size(0), adj.size(0)).to(self.device)
            out[i,j] = gumbel_output.float()
            out.T[i,j] = gumbel_output.float()

        else:
            out = torch.zeros(adj.size(0), adj.size(0)).to(self.device)
            out[i,j] = gumbel_output[:,0]
            out.T[i,j] = gumbel_output[:,0]
            
        out = out.to(self.device)
        out = out + torch.eye(out.shape[0]).to(self.device)
        return out
    
    '''
    gumbel softmax - to sample both soft and hard
    '''
    def soft_hard_gumbel_softmax(self, logits, temperature):
        y = self.gumbel_softmax_sample(logits, temperature)
        y_hard = torch.max(y.data, 1)[1]
        return y, y_hard

    def soft_hard_sample(self, adj, temperature):
        i,j = torch.triu_indices(adj.size(0), adj.size(0), offset=1)
        prob_ex = adj[i,j]
        prob_unex = abs(1-prob_ex)
        gumbel_input = torch.stack((prob_ex, prob_unex), dim=1)
        gumbel_output_soft, gumbel_output_hard = self.soft_hard_gumbel_softmax(gumbel_input, temperature)

        soft_out = torch.zeros(adj.size(0), adj.size(0)).to(self.device)
        soft_out[i,j] = gumbel_output_soft[:,0]
        soft_out.T[i,j] = gumbel_output_soft[:,0]

        soft_out = soft_out.to(self.device)
        soft_out = soft_out + torch.eye(soft_out.shape[0]).to(self.device)

        hard_out = torch.zeros(adj.size(0), adj.size(0)).to(self.device)
        hard_out[i,j] = 1-gumbel_output_hard.float()
        hard_out.T[i,j] = 1-gumbel_output_hard.float()
            
        hard_out = hard_out.to(self.device)
        hard_out = hard_out + torch.eye(hard_out.shape[0]).to(self.device)

        return soft_out, hard_out



