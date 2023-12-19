from torch_geometric.nn import GCNConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv1d):
            print(m)
            m.weight.data.normal_(0, 0.02)
    
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
            
def global_mean_pool(x: Tensor, batch: Optional[Tensor],
                     size: Optional[int] = None) -> Tensor:
    if batch is None:
        return x.mean(dim=-2, keepdim=x.dim() == 2)
    size = int(batch.max().item() + 1) if size is None else size
    return scatter(x, batch, dim=-2, dim_size=size, reduce='mean')




# Spatial GNN for ISRUC 
class SpatialGNN_Sleep(torch.nn.Module):
    def __init__(
        self, 
        num_nodes: int=6,
        in_channels:int=3000, 
        out_channels:int=128, 
        kernel_size:int=25,
        stride: int = 3,
        dropout: float = 0.35,
        add_self_loops:bool=True,
        device: str='cuda',
        mode : str = 'sequence_wise',
        drop_edge : bool = True
    ):
        super(SpatialGNN_Sleep, self).__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.add_self_loops = add_self_loops
        self.stride = stride
        self.dropout = dropout
        self.Conv1D = nn.Conv1d(in_channels=num_nodes, out_channels=num_nodes, kernel_size=kernel_size).to(device)
        self.GCN = GCNConv(in_channels=65, out_channels=out_channels, add_self_loops=add_self_loops).to(device)

        self.conv_block1 = nn.Sequential(
                nn.Conv1d(num_nodes, 6*num_nodes, kernel_size=self.kernel_size,
                        stride=self.stride, bias=False, padding = (self.kernel_size//2), groups = self.num_nodes),
                nn.BatchNorm1d(6*num_nodes),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
                nn.Dropout(self.dropout)
            )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(6*num_nodes, 16*num_nodes, kernel_size=8, stride=1, bias=False, padding=4, groups = self.num_nodes),
            nn.BatchNorm1d(16*num_nodes),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(16*num_nodes, num_nodes, kernel_size=8, stride=1, bias=False, padding=4, groups = self.num_nodes),
            nn.BatchNorm1d(num_nodes),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.lin1 = torch.nn.Linear(self.in_channels, self.out_channels).to(device) # har
        self.mode = mode
        self.drop_edge = drop_edge

        self.fft_lin = nn.Linear(3000,1501)

        self.output_layer = nn.Linear(self.out_channels, 5)
        
        initialize_weights(self)

    def forward(
        self, X: torch.FloatTensor, edge_index: torch.LongTensor, edge_weight : torch.FloatTensor = None, drop_edge : bool = False, signal='fft'
    ) -> torch.FloatTensor:
        if signal == 'raw':
            X = self.fft_lin(X)

        x = self.conv_block1(X)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        T = torch.zeros(x.size(0), x.size(1), self.out_channels).to(x.device)
        # [1, batch(# epochs), # channel, # feature]

        if edge_weight != None:
            if self.mode == 'sequence_wise':
                for t in range(x.size(0)): # epochs
                    if drop_edge:
                        T[t] = self.GCN(x[t], edge_index[t], edge_weight[t]).to(X.device)
                    else:
                        T[t] = self.GCN(x[t], edge_index, edge_weight[t]).to(X.device)

            elif self.mode == 'channel_wise':
                for t in range(x.size(0)): #  epochs
                    if drop_edge:
                        T[t] = self.GCN(x[t], edge_index[t], edge_weight).to(X.device)
                    else:                 
                        T[t] = self.GCN(x[t], edge_index, edge_weight).to(X.device)
        else:
            for t in range(x.size(0)): #  epochs
                if drop_edge:
                    T[t] = self.GCN(x[t], edge_index[t], edge_weight).to(X.device)
                else: 
                    T[t] = self.GCN(x[t], edge_index, edge_weight).to(X.device)

        T = F.elu(T+self.lin1(X))
        output = global_mean_pool(T, batch=None)

        class_output = self.output_layer(output)
        return output, class_output


## Spatial GNN for HAR 
class SpatialGNN_HAR(nn.Module):
    def __init__(
        self, 
        num_nodes: int=6,
        in_channels:int=3000, 
        out_channels:int=256, 
        kernel_size:int=8,
        stride: int = 1,
        dropout: float = 0.35,
        add_self_loops:bool=True,
        device: str='cuda',
        mode : str = 'sequence_wise',
        drop_edge : bool = True,
        # is_pairnorm : bool = False,
        # pairnorm_scale : float = 1.0
    ):
        super(SpatialGNN_HAR, self).__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.add_self_loops = add_self_loops
        self.stride = stride
        self.dropout = dropout
        # self.is_pairnorm = is_pairnorm
        # self.pairnorm_scale = pairnorm_scale
        self.GCN = GCNConv(in_channels=18, out_channels=self.out_channels, add_self_loops=add_self_loops).to(device)


        self.conv_block1 = nn.Sequential(
            nn.Conv1d(num_nodes, 32, kernel_size=self.kernel_size,
                      stride=self.stride, bias=False, padding = (self.kernel_size//2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(self.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, 9, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(9),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.lin1 = torch.nn.Linear(self.in_channels, self.out_channels).to(device) # har
        # self.pairnorm = PairNorm('PN', self.pairnorm_scale)
        self.mode = mode
        self.drop_edge = drop_edge

        self.fft_lin = nn.Linear(65,128)

        self.output_layer = nn.Linear(self.out_channels, 6)

        initialize_weights(self)

    def forward(
        self, X: torch.FloatTensor, edge_index: torch.LongTensor, 
        edge_weight : torch.FloatTensor = None, drop_edge : bool = False, signal='raw'
    ) -> torch.FloatTensor:
        
        if signal=='fft':
            X=self.fft_lin(X)


        #### Convolution #######
        x = self.conv_block1(X)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        T = torch.zeros(x.size(0), x.size(1), self.out_channels).to(x.device)
        # [1, batch(# epochs), # channel, # feature]

        if edge_weight != None:
            if self.mode == 'sequence_wise':
                for t in range(x.size(0)): # epochs
                    if drop_edge:
                        T[t] = self.GCN(x[t], edge_index[t], edge_weight[t]).to(X.device)
                    else:
                        T[t] = self.GCN(x[t], edge_index, edge_weight[t]).to(X.device)

            elif self.mode == 'channel_wise':
                for t in range(x.size(0)): #  epochs
                    if drop_edge:
                        T[t] = self.GCN(x[t], edge_index[t], edge_weight).to(X.device)
                    else:                 
                        T[t] = self.GCN(x[t], edge_index, edge_weight).to(X.device)

            elif self.mode == 'edge_mask':
                for t in range(x.size(0)):
                    T[t] = self.GCN(x[t], edge_index, edge_weight[t]).to(X.device)
        else:
            for t in range(x.size(0)): #  epochs
                if drop_edge:
                    T[t] = self.GCN(x[t], edge_index[t], edge_weight).to(X.device)
                else: 
                    T[t] = self.GCN(x[t], edge_index, edge_weight).to(X.device)
        
        # if self.is_pairnorm:
        #     T = self.pairnorm(T)

        T = F.elu(T + self.lin1(X))

        output = global_mean_pool(T, batch=None)
        
        class_output = self.output_layer(output)
        return output, class_output


    
