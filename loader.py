from numpy.core.fromnumeric import transpose

from torch.utils.data import DataLoader
from torch.utils.data import Dataset 
from torch.utils.data import TensorDataset 

import os
import numpy as np
import glob
import torch
import utils  
from utils import std_by_channel, fft



class Data(Dataset):
    def __init__(self, root_dir, sensor_dir, normalization=False, channels=['C3-M2','C4-M1','F3-M2','F4-M1','O1-M2','O2-M1'], n_sub = 3):

        # data path  
        self.root_dir = root_dir 
        self.sensor_dir = sensor_dir     
        sensor_path = os.path.join(root_dir, sensor_dir)
        self.file_names = glob.glob(os.path.join(sensor_path,'*.npz'))
        self.normalization = normalization
        self.channels = channels


    def __len__(self):
        return len(self.file_names)

        
    def __getitem__(self, idx):
        file = np.load(self.file_names[idx])
        f_name = self.file_names[idx]

        x = []
        x = ([file[ch] for ch in self.channels])
        fft_x= ([fft(file[ch]) for ch in self.channels])

        data = np.stack(x, axis=0)
        fft_data = np.stack(fft_x, axis=0)

        target = file['y']

        if data.ndim==4:
            data = data.squeeze(-1)

        if self.normalization:
            data = std_by_channel(data) 
            fft_data = std_by_channel(fft_data)

        return data.transpose(1,0,2), fft_data.transpose(1,0,2),target.reshape(-1)


def dataloader(root_dir, sensor_dir, batch_size, normalization, channels, n_sub = 3):

    dataset = Data(root_dir = root_dir, sensor_dir = sensor_dir, normalization = normalization, channels = channels, n_sub = n_sub)
    

    file_load = DataLoader(dataset = dataset, batch_size = 1, shuffle = True)
    

    return file_load    
    
    

def batch_loader(data, fft, y, batch_size):
    dataset = TensorDataset(data, fft, y)
    dataloader = DataLoader(dataset, batch_size= batch_size, shuffle=True, drop_last = True)
    return dataloader



# for batch learning
def graph_loader(data, batch_size):
    # load into graph data
    loader = DataLoader(dataset = data, batch_size=batch_size, shuffle=False)
    return loader 

class Batch_Dataset(Dataset):
    def __init__(self, x, y):
        super(Batch_Dataset, self).__init__()

        self.x = x
        self.y = y
        self.len = x.shape[0]


    def __getitem__(self, index):

        return self.x[index], self.y[index]

    def __len__(self):
        return self.len

def batchloader(x,y, batch_size):
    dataset = Batch_Dataset(x,y)
    data_loader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle=True)

    return data_loader
