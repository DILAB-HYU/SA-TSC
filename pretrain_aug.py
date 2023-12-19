import gc, os , yaml

import numpy as np

import loader
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from spatial_TCC import SpatialGNN_Sleep, SpatialGNN_HAR 

from ssl_loss import Barlow_loss

from utils import loss_plot, batched_random_drop_edge, batched_sim_matrix, addNoise
from utils import construct_adjacency, sim_matrix

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.nn import dense_mincut_pool
from torch_geometric.utils.dropout import dropout_adj 

from gumbel_generate import Gumbel

from time import sleep

'''
spatial gnn pre-train using SSL Loss (Barlow twins)
'''
torch.autograd.set_detect_anomaly(True)
class Graph_sleep(object):
    def __init__(self, args):
        with open('config.yaml', "r") as ymlfile:
            cfg = yaml.full_load(ymlfile)
        ########## PARAMETER SETTING ##########

        # training parameters
        self.epoch = args.epoch
        self.device = args.device

        # data dir 
        self.data_set = args.data_set
        self.root_dir = args.root_dir
        self.batch_size = args.batch_size
        # save info 
        ## model saved at "save_dir / model_name"
        self.model_name = args.model_name
        self.exp_name = args.exp_name
        self.save_dir = args.save_dir
        self.save_mode = args.save
        self.print_interval = args.print_interval
        self.save_root = os.path.join(self.save_dir, self.model_name, self.data_set) #[save_dir] / [model_name] / [data_set]]
        
        # parameter for model training
        ## st_model
        self.distance = args.ssl_distance_adj
        self.randomness = args.randomness

        ## spectral_clustering
        self.gumbel = args.gumbel
        self.adj_dist = args.adj_dist
        self.gumbel_only = args.gumbel_only

        self.channel_name  = cfg[self.data_set]['channel_name']
        self.n_cluster = cfg[self.data_set]['cluster_num']

        self.spatio_edge = cfg[self.data_set]['spatio_edge']
        self.spatio_edge = torch.tensor(self.spatio_edge, dtype=torch.long).t().contiguous()

        self.spatio_edge_2 = cfg[self.data_set]['spatio_edge']
        self.spatio_edge_2 = torch.tensor(self.spatio_edge_2, dtype=torch.long)
        
        self.num_nodes = cfg[self.data_set]['num_nodes']
        self.input_dim = cfg[self.data_set]['input_dim'] # fft_dim


        self.output_dim = args.output_dim
        self.kernel_size = args.kernel_size
        self.hidden_dim = self.output_dim # out dim of spatial
        self.fft_dim = cfg[self.data_set]['fft_dim']

        self.aug_mode = args.aug_mode 

        self.use_gumbel = args.gumbel
        self.gumbel_tmp = args.gumbel_tmp

        ########## Data Load ##########
        print('Dataset: ', self.data_set)
        
        # load each subject 
        self.file_loader = loader.dataloader(root_dir = os.path.join(self.root_dir, self.data_set), sensor_dir = 'train', batch_size=1, normalization=True, channels = self.channel_name)

        ########## NETWORK INIT ##########
        print(self.device)
        if self.data_set == 'ISRUC' or self.data_set == 'SleepEDF':
            self.spatio_model =  SpatialGNN_Sleep(self.num_nodes, self.fft_dim, self.output_dim, self.kernel_size, device=self.device, mode='sequence_wise')
        elif self.data_set == 'HAR':
            self.spatio_model = SpatialGNN_HAR(self.num_nodes, self.input_dim, self.output_dim, self.kernel_size, device=self.device, mode='sequence_wise')

        self.barlow = Barlow_loss(out_dim = self.hidden_dim).to(self.device)
        self.gumbel = Gumbel(self.device)
        self.dropout = nn.Dropout(p=0.2)


        self.adj_dist = args.adj_dist
        self.adj_norm = args.adj_norm
        self.mu = args.mu

        self.alpha = args.alpha
        self.beta = args.beta

        self.st_optimizer = torch.optim.Adam([{'params':self.spatio_model.parameters()}], lr = args.lr)

        self.spatio_model = self.spatio_model.to(self.device)
        self.spatio_edge = self.spatio_edge.to(self.device)
        self.spatio_edge_2 = self.spatio_edge_2.to(self.device)

        print('---------- Networks architecture -------------')
        print(self.spatio_model,'\n')
        print('-----------------------------------------------')
        


    def train(self):
        self.train_hist = {}
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        self.spatio_model.train()
        writer = SummaryWriter()

        print('training start!!')

        print("Running training...")

        print('Total Subject #', len(self.file_loader))
        for epoch in range(self.epoch):
            step = 0
            total_ssl_loss  = 0, 0, 0 
            losses, ssl_losses, mc_losses, o_losses, agr_losses = [], [], [], [], []

            for sub_iter, (raw, fft, target) in enumerate(tqdm(self.file_loader)):
                raw,fft,target = raw.squeeze(0), fft.squeeze(0), target.squeeze(0)
                batchloader = loader.batch_loader(raw, fft, target, self.batch_size)
                for iter, (x, x_fft, target) in enumerate((batchloader)):
                    x = x.type(torch.FloatTensor).to(self.device)
                    x_fft = x_fft.type(torch.FloatTensor).to(self.device)
                    target = target.type(torch.LongTensor).to(self.device)  

                    random_drop_edge = batched_random_drop_edge(x.size(0), x.size(1), self.spatio_edge_2, ratio=0.2, device = self.device) ## only for random drop edges. 

                    self.st_optimizer.zero_grad()
                    
                    if self.data_set == 'ISRUC' or self.data_set == 'SleepEDF':
                        spatial_output, s = self.spatio_model(x_fft, edge_index = self.spatio_edge) # [1 (Batch), num_epoch, 256] 

                        ### node feature augmentation (NOT "NODE" augmentation) 
                        if self.aug_mode == 'add_noise':  
                            x_fft_noise = addNoise(x_fft)
                            spatial_output_2, s = self.spatio_model(x_fft_noise, edge_index = self.spatio_edge)

                        elif self.aug_mode == 'feature_drop':
                            x_fft_drop = self.dropout(x_fft)

                            spatial_output_2, s = self.spatio_model(x_fft_drop, edge_index = self.spatio_edge) 
                        
                        elif self.aug_mode == 'edge_mask':
                            random_drop_edge, _ = dropout_adj(self.spatio_edge, p=0.2, force_undirected=True, num_nodes = self.num_nodes)                     
                            sim_out = sim_matrix(x_fft.squeeze(), random_drop_edge.t())
                            spatial_output_2, s = self.spatio_model(x_fft, edge_index = random_drop_edge, edge_weight = sim_out.to(self.device), drop_edge=False)
                
                        elif self.aug_mode == 'cross_domain':
                            spatial_output, s = self.spatio_model(x_fft, edge_index = self.spatio_edge) 

                            if self.randomness:
                                sim_out = batched_sim_matrix(x, random_drop_edge.transpose(1,2), metric=self.distance, device= self.device) 
                                spatial_output_2, s_2 = self.spatio_model(x_fft, edge_index = random_drop_edge, edge_weight = sim_out.to(self.device),  drop_edge=True) 
                            else:
                                #### DEFAULT #### 
                                spatial_output_2, s_2 = self.spatio_model(x, edge_index = self.spatio_edge, signal = 'raw')   
                        
                        elif self.aug_mode == 'distance':
                            spatial_output, s = self.spatio_model(x_fft, edge_index = self.spatio_edge)
                            sim_out = batched_sim_matrix(x, random_drop_edge.transpose(1,2), metric=self.distance, device= self.device) 
                            spatial_output_2, s_2 = self.spatio_model(x_fft, edge_index = random_drop_edge, edge_weight = sim_out.to(self.device),  drop_edge=True) 


                    else: #### "HAR"
                        spatial_output, s = self.spatio_model(x, edge_index = self.spatio_edge) 
                        
                        ### node feature augmentation (NOT "NODE" augmentation) 
                        if self.aug_mode == 'add_noise':  
                            x_noise = addNoise(x)
                            spatial_output_2, s = self.spatio_model(x_noise, edge_index = self.spatio_edge)

                        elif self.aug_mode == 'feature_drop':
                            x_drop = self.dropout(x)

                            spatial_output_2, s = self.spatio_model(x_drop, edge_index = self.spatio_edge) # [1 (Batch), num_epoch, 256]
                        
                        elif self.aug_mode == 'edge_mask':
                            
                            random_drop_edge, _ = dropout_adj(self.spatio_edge, p=0.2, force_undirected=True, num_nodes = self.num_nodes)                     
                            sim_out = sim_matrix(x_fft.squeeze(), random_drop_edge.t())
                            spatial_output_2, s = self.spatio_model(x, edge_index = random_drop_edge, edge_weight = sim_out.to(self.device), drop_edge=False)
                
                        elif self.aug_mode == 'cross_domain':
                            spatial_output, s = self.spatio_model(x, edge_index = self.spatio_edge) # [1 (Batch), num_epoch, 256] 

                            if self.randomness: # edge drop
                                sim_out = batched_sim_matrix(x_fft, random_drop_edge.transpose(1,2), metric=self.distance, device= self.device) 
                                spatial_output_2, s_2 = self.spatio_model(x, edge_index = random_drop_edge, edge_weight = sim_out.to(self.device),  drop_edge=True)
                            else:
                                #### DEFAULT #### 
                                spatial_output_2, s_2 = self.spatio_model(x_fft, edge_index = self.spatio_edge, signal = 'fft')  
                        
                        elif self.aug_mode == 'distance':
                            spatial_output, s = self.spatio_model(x, edge_index = self.spatio_edge) # [1 (Batch), num_epoch, 256] 
                            sim_out = batched_sim_matrix(x_fft, random_drop_edge.transpose(1,2), metric=self.distance, device= self.device)
                            spatial_output_2, s_2 = self.spatio_model(x, edge_index = random_drop_edge, edge_weight = sim_out.to(self.device),  drop_edge=True)  

                    '''
                    barlow loss
                    '''
                    on_diag, off_diag = self.barlow(spatial_output, spatial_output_2)
                    ssl_loss = (on_diag + 0.005 * off_diag ) 

                    feature = spatial_output
                    adj = construct_adjacency(feature.squeeze(0), distance = self.adj_dist, mu_val=self.mu, adj_norm = self.adj_norm)

                    adj = adj.squeeze().detach()
                    g_adj = adj.to(self.device)
                    if self.use_gumbel:
                        if self.gumbel_only : 
                             g_adj = self.gumbel.soft_hard_sample(adj = g_adj, temperature=self.gumbel_tmp)[0] 
                        else:
                            g_adj = g_adj * self.gumbel.soft_hard_sample(adj = g_adj, temperature=self.gumbel_tmp)[0] # gumbel = hard False

                    hard_adj = self.gumbel.soft_hard_sample(adj = g_adj, temperature=self.gumbel_tmp)[1]

                    '''
                    agreement  loss
                    '''
                    eps = 1e-5
                    agreement_loss = 0
                    for idx, (spatial, class_s) in enumerate(zip(spatial_output.detach(), s)):
                        rep_error = spatial-spatial_output
                        class_error = class_s-s

                        h_agreement = torch.sum(hard_adj[idx][:,None]*rep_error)/sum(hard_adj[idx]) # to minimize
                        h_disagreement = torch.sum((1-hard_adj[idx][:,None])*rep_error)/sum(1-hard_adj[idx]) # to maximize
                        h_loss = h_agreement/h_disagreement
                        
                        s_agreement = torch.sum(hard_adj[idx][:,None]*class_error)/sum(hard_adj[idx]) # to minimize
                        s_disagreement = torch.sum((1-hard_adj[idx][:,None])*class_error)/sum(1-hard_adj[idx]) # to maximize
                        s_loss = s_agreement/(s_disagreement+eps)
                        
                        agreement_loss += h_loss * s_loss
                    agreement_loss = agreement_loss/idx
                    

                    if epoch < 50:
                        loss = ssl_loss + self.alpha * (agreement_loss) 
                        losses.append(loss.item())
                        ssl_losses.append(ssl_loss.item())
                        agr_losses.append(agreement_loss.item())

                    else: 
                        _, _, mc_loss, o_loss = dense_mincut_pool(feature, g_adj, s)
                        loss = ssl_loss + self.alpha * (agreement_loss) + self.beta * (mc_loss + o_loss)

                        losses.append(loss.item())
                        ssl_losses.append(ssl_loss.item())
                        mc_losses.append(mc_loss.item())
                        o_losses.append(o_loss.item())
                        agr_losses.append(agreement_loss.item())

                    

                    loss.backward()
                    self.st_optimizer.step()

            writer.add_scalar('loss/barlow_loss', np.average(ssl_losses), iter)
            writer.add_scalar('loss/mc_loss', np.average(mc_losses), iter)
            writer.add_scalar('loss/o_loss', np.average(o_losses), iter)
            writer.add_scalar('loss/total_loss', np.average(losses), iter)

            if epoch < 50:
                print(f'Epoch {epoch+1} / {self.epoch}, Subject {sub_iter+1}/{len(self.file_loader)}, step {iter+1}/{len(batchloader)}, total loss = {np.average(losses):.4f}, ssl_loss = {np.average(ssl_losses):.4f}, agree_loss = {np.average(agr_losses):.4f}')
            else:
                print(f'Epoch {epoch+1} / {self.epoch}, Subject {sub_iter+1}/{len(self.file_loader)}, step {iter+1}/{len(batchloader)}, total loss = {np.average(losses):.4f}, ssl_loss = {np.average(ssl_losses):.4f}, mc_loss = {np.average(mc_losses):.4f},  o_loss = {np.average(o_losses):.4f}, agree_loss = {np.average(agr_losses):.4f}')

        

            file_name = str(self.output_dim)+str(self.kernel_size)
                
            if (epoch+1) % 10 == 0:
                if self.save_mode:   
                    if not os.path.exists(os.path.join(self.save_root, self.exp_name, file_name)):
                        os.makedirs(os.path.join(self.save_root, self.exp_name, file_name))
                    torch.save(self.spatio_model.state_dict(), os.path.join(self.save_root, self.exp_name, file_name, 'spatio_{}epoch.pth'.format((epoch+1))))
                    
        print("Training finish!")
