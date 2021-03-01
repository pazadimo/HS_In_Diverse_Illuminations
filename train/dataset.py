import torch.utils.data as data
import torch
import h5py
import numpy as np
from utils import save_matv73

class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        hf = h5py.File(file_path)
        self.data = hf.get('data')
        self.target = hf.get('label')

    def __getitem__(self, index):
        # print(np.shape(self.data[index,:,:,:]))
        # save_matv73('./input30' +str(index)+ '.mat', 'rad',self.data[index,:,:,:])
        # save_matv73('./label30' +str(index)+ '.mat', 'rad',self.target[index,:,:,:])
        # return torch.from_numpy(self.data[index,11:16,:,:]).float(), torch.from_numpy(self.target[index,22,:,:]).float()
        # DATA = self.data[index,:,:,:] * 20.0000  / 6.0000
        # TARGET = self.target[index,:,:,:] * 20.0000  / 6.0000
        DATA = self.data[index,:,:,:] * 20.0000  / 6.0000
        TARGET = self.target[index,:,:,:] * 20.0000  / 6.0000
        return torch.from_numpy(DATA).float(), torch.from_numpy(TARGET).float()
        
    def __len__(self):
        return self.data.shape[0]
