import torch
import math, os
import yaml, glob
from spatial_TCC_v3 import SpatialGNN_Sleep, SpatialGNN_HAR 

from torch.utils.data import  DataLoader
import torch.nn as nn
from torch.nn import Linear
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from utils import loss_plot, batched_random_drop_edge, batched_sim_matrix
import loader
from utils import initialize_weights


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

class Finetune(object):
    def __init__(self, args):
        with open('config.yaml','r') as ymlfile:
            cfg = yaml.full_load(ymlfile)
        
        self.seed = args.random_seed
        seed_everything(self.seed)
        self.mode = args.mode
        self.root_dir = args.root_dir
        self.data_set = args.data_set
        self.device = args.device
        self.save_dir = 'saved_models'
        self.model_name = args.model_name
        self.exp_name = args.exp_name
        self.n_epochs = args.epoch
        self.batch_size = args.batch_size

        # Dataset configuration
        self.n_cluster = cfg[self.data_set]['cluster_num']
        self.channel_name  = cfg[self.data_set]['channel_name']
        self.num_nodes = cfg[self.data_set]['num_nodes']
        self.input_dim = cfg[self.data_set]['input_dim']
        self.fft_dim = cfg[self.data_set]['fft_dim']

        self.spatio_edge = cfg[self.data_set]['spatio_edge']
        self.spatio_edge = torch.tensor(self.spatio_edge, dtype=torch.long).t().contiguous().to(self.device)

        self.spatio_edge_2 = cfg[self.data_set]['spatio_edge']
        self.spatio_edge_2 = torch.tensor(self.spatio_edge_2, dtype=torch.long)
        
        self.distance = 'gaussian_kernel'

        self.file_name = args.file_name
        if self.data_set == 'ISRUC' or self.data_set == 'SleepEDF':
            self.idx = -2
        elif self.data_set == 'HAR':
            self.idx = -1
        self.output_dim = int(self.file_name[:self.idx])
        self.hidden_dim = self.output_dim
        self.kernel_size = int(self.file_name[self.idx:])
        self.criterion  = nn.CrossEntropyLoss().to(self.device)

        self.output_dim = int(self.file_name[:-1])
        self.kernel_size= int(self.file_name[-1:])
        # Model & Saved model file
        if self.data_set == 'ISRUC' or self.data_set == 'SleepEDF':
            self.spatio =  SpatialGNN_Sleep(self.num_nodes, self.fft_dim, self.output_dim, self.kernel_size, device=self.device, mode='sequence_wise')
        elif self.data_set == 'HAR':
            self.spatio = SpatialGNN_HAR(self.num_nodes, self.input_dim, self.output_dim, self.kernel_size, device=self.device, mode='sequence_wise')
       
        print(self.spatio)
        self.spt_pth_name = 'spatio_' + str(args.file_epoch) +'epoch.pth'
        print(os.path.join(self.save_dir, self.model_name, self.data_set, self.exp_name, self.file_name, self.spt_pth_name))
        self.spatio_file = glob.glob(os.path.join(self.save_dir, self.model_name, self.data_set, self.exp_name, self.file_name, self.spt_pth_name)).pop()
        self.spatio.load_state_dict(torch.load(self.spatio_file))
        self.spatio = self.spatio.to(self.device)

        print('='*80)
        print('Dataset', self.data_set)
        print('='*80)
        print('spatio file: ', self.spatio_file)
        print('='*80)
        
        self.seed = args.random_seed
        seed_everything(self.seed)

        # Data Loader & Train/Test split for finetuning
        self.train_loader = loader.dataloader(root_dir = os.path.join(self.root_dir, self.data_set), sensor_dir = 'ft_'+ str(args.subject)+'sub', #3
                        batch_size=1, normalization=True, channels = self.channel_name)
        self.test_loader = loader.dataloader(root_dir = os.path.join(self.root_dir, self.data_set), sensor_dir = 'test', 
                        batch_size=1, normalization=True, channels = self.channel_name)

        print('Train Set, Test Set : ', len(self.train_loader),'/', len(self.test_loader))


        # optimizer
        self.optimizer = torch.optim.AdamW([{'params':self.spatio.parameters()}], lr = args.spt_lr, betas=(0.9, 0.98))

         # gradient backward flow -> False
        if self.mode == 'finetune':
            for param in self.spatio.parameters():
                param.requires_grad = True
                
        elif self.mode == 'linear_eval':
            for param in self.spatio.parameters():
                param.requires_grad = False
            self.spatio.output_layer.weight.requires_grad =True
            self.spatio.output_layer.bias.requires_grad=True

    def train(self):
        print('Finetuning Training Start')
        print('**********Batch Learning*********')

        self.spatio.train()
        n_total_steps = len(self.train_loader)

        for epoch in range(self.n_epochs):
            for iter, (x, fft, target) in enumerate(self.train_loader):
                x = x.type(torch.FloatTensor).squeeze(0).to(self.device)
                x_fft = fft.type(torch.FloatTensor).squeeze(0).to(self.device)
                target = target.type(torch.LongTensor).squeeze(0).to(self.device)  
                if self.data_set == 'ISRUC' or self.data_set == 'SleepEDF':
                    batch_loader = loader.batchloader(x_fft,target, batch_size = self.batch_size)
                elif self.data_set == 'HAR':
                    batch_loader = loader.batchloader(x,target, batch_size = self.batch_size)

                for (b_x,b_target) in batch_loader:
                    self.optimizer.zero_grad()
                    spatial_output, pred = self.spatio(b_x, edge_index = self.spatio_edge) # [1 (Batch), num_epoch, 256] 

                    pred = pred.squeeze(0)
                    target = b_target.squeeze(0)

                    loss = self.criterion(pred, target) 
                    loss.backward()
                    self.optimizer.step()

                if (iter+1) % 1 == 0:
                    print(f'epoch {epoch+1} / {self.n_epochs}, step {iter+1}/{n_total_steps}, loss = {loss.item():.4f}')
                
    def test(self):
        print('Final Prediction Start')
        with torch.no_grad():
            self.spatio.eval()

            predlist=torch.zeros(0,dtype=torch.long, device='cpu')
            lbllist=torch.zeros(0,dtype=torch.long, device='cpu')

            for _, (x, fft, target) in enumerate(self.test_loader): 

                x = x.type(torch.FloatTensor).squeeze(0).to(self.device)
                x_fft = fft.type(torch.FloatTensor).squeeze(0).to(self.device)
                target = target.type(torch.LongTensor).squeeze(0).to(self.device)  

                self.optimizer.zero_grad()
                if self.data_set == 'ISRUC' or self.data_set == 'SleepEDF':
                    batch_loader = loader.batchloader(x_fft,target, batch_size = self.batch_size)
                elif self.data_set == 'HAR':
                    batch_loader = loader.batchloader(x,target, batch_size = self.batch_size)                
                    
                for (b_x,b_target) in batch_loader:

                    spatial_output, pred = self.spatio(b_x, edge_index = self.spatio_edge) # [1 (Batch), num_epoch, 256]
                    _, predictions = torch.max(pred,-1)


                    predlist = torch.cat([predlist, predictions.view(-1).cpu()])
                    lbllist = torch.cat([lbllist, b_target.view(-1).cpu()])

            acc = accuracy_score(lbllist, predlist)
            f1 = f1_score(lbllist, predlist, average='macro')
            class_f1 = f1_score(lbllist, predlist, average=None)

            print("=============================")
            print(predlist)
            print("=============================")
            print(lbllist)
            print("=============================")

            report = classification_report(lbllist, predlist, output_dict = False)
            print(report)
            print(confusion_matrix(lbllist, predlist))
            print('acc: ', acc, 'f1: ', f1)
        return acc, f1, class_f1

