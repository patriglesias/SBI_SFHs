
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
from accelerate import Accelerator #to use pytorch
from torch.utils.data import DataLoader
import spender
from ..generate_input import *


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

#generate data:

#generate parametrizations
t,ms,percentiles=generate_weights_from_SFHs(SFR=sfr_linear_exp,mgal=10**10,tau=np.linspace(0.3,5,100),ti=np.arange(0,5,0.5),tmin=0,tmax=14,step=0.01,percen=True)
#load MILES spectra and interpolate
wave,data=get_data(dir_name='../MILES_BASTI_KU_baseFe',strs_1='Mku1.30Zp0.06T',strs_2='_iTp0.00_baseFe.fits')
tbins=get_tbins(dir_name='../MILES_BASTI_KU_baseFe',strs_1='Mku1.30Zp0.06T',strs_2='_iTp0.00_baseFe.fits')
data_extended=interpolate(tbins,t,data)
#generate spectra for the parametrized SFHs
wave,seds=generate_all_spectrums(t,ms,wave,data_extended)

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
params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6}
max_epochs = 100

# Datasets 
#seds shape(1000, 10)
#percentiles shape (1000, 4300)

x_train = seds[:,:] #seds
y_train = percentiles[:,:] #percentiles


"""x_train = seds[:int(0.8*len(seds)),:] #seds
y_train = percentiles[:int(0.8*len(seds)),:] #percentiles

x_val = seds[int(0.8*len(seds)):int(0.9*len(seds)),:] #seds
y_val = percentiles[int(0.8*len(seds)):int(0.9*len(seds)),:] #percentiles

x_test = seds[int(0.9*len(seds)):,:] #seds
y_test = percentiles[int(0.9*len(seds)):,:] #percentiles"""



# Generators
training_set = Dataset(x_train, y_train)
training_generator = torch.utils.data.DataLoader(training_set, **params)

"""
validation_set = Dataset(x_val, y_val)
validation_generator = torch.utils.data.DataLoader(validation_set, **params)"""


dataloader = Accelerator.prepare(training_generator)

"""ss, losses, halphas, zs, norms, ids = [], [], [], [], [], []
with torch.no_grad():

    for spec, w, z, id, norm, zerr in dataloader:
        # need the latents, of course
        s = model.encode(spec, aux=z.unsqueeze(1))
        # everything else is to color code the UMap
        s, spec_rest, spec_reco = model._forward(spec, z=z, s=s) # reuse latents
        loss = model._loss(spec, w, spec_reco, individual=True)
        halpha = l_halpha(model, spec_reco, spec_rest, dim=-1)
        
        ss.append(s)
        losses.append(loss)
        halphas.append(halpha)
        zs.append(z)
        norms.append(norm)
        ids.append(id)

ss = np.concatenate(ss, axis=0) # converts to numpy
losses = np.concatenate(losses, axis=0)
halphas = np.concatenate(halphas, axis=0)
zs = np.concatenate(zs, axis=0)
norms = np.concatenate(norms, axis=0)
ids = np.concatenate(ids, axis=0)"""

