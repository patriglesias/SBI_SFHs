#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 12:51:10 2023

@author: patriglesiasnavarro
"""

from torchviz import make_dot

from zlib import Z_SYNC_FLUSH
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm,trange
import torch
from torch import nn
from torch import optim
from accelerate import Accelerator #to use pytorch
from torch.utils.data import DataLoader
from spender import SpectrumEncoder,MLP,encoder_percentiles,load_model

print('Modules prepared')


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
#torch.cuda.set_device(1)
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
print("GPU" if use_cuda else "CPU",' prepared')




class Dataset(torch.utils.data.Dataset):

    def __init__(self,x,y):

        """ generate and organize artificial data from parametrizations of SFHs"""

        self.x=torch.from_numpy(x) #seds
        self.y=torch.from_numpy(y) #percentiles


    def __len__(self):
        """total number of samples"""
        return len(self.x[:,0])

    def __getitem__(self,index):
        """Generates one sample of data"""
        x=self.x[index,:]
        y=self.y[index,:]
        return x,y


# Parameters
batch_size=128
max_epochs=250
lr=5e-4
params = {'batch_size': batch_size,
          'shuffle': True}
n_latent=16




x_train = torch.autograd.Variable(torch.ones(1,4300),requires_grad=True) # fake seds


### TRAINING MODE ###


model = encoder_percentiles(n_latent=n_latent,n_out=11,n_hidden=(16,32),act=None,dropout_2=0.0)

latents=model(x_train)

#torchviz
#make_dot(latents[0],params=dict(model.named_parameters())).render("attached", format="png")

#torchview

from torchview import draw_graph


batch_size = 128
# device='meta' -> no memory is consumed for visualization
model_graph = draw_graph(model, input_size=(batch_size, 4300), device='meta')
model_graph.visual_graph

