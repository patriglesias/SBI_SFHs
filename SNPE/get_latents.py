### Load model and get a dataset of spectra - latents - percentiles ###


from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
from accelerate import Accelerator #to use pytorch
from torch.utils.data import DataLoader
from ..spender.spender import SpectrumEncoder,MLP,encoder_percentiles,load_model




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

#load data:
percentiles=np.load('../spender/saved_input/percentiles.npy')
wave=np.load('../spender/saved_input/waves.npy')
seds=np.load('../../seds.npy')
#ms=np.load('../../sfh.npy') #s

#create a dataset
dataset = Dataset(seds, percentiles)
print('Shape of the dataset: ',np.shape(seds))
params={'batch_size': 512 } 
generator = torch.utils.data.DataLoader(dataset,**params) #with minibatches

#call accelerator
accelerator = Accelerator(mixed_precision='fp16')
loader = accelerator.prepare(generator)

#load model
n_latent=16
model_file = "../spender/saved_model/generate_latent_2/latent_"+str(n_latent)+"/checkpoint.pt"
model, loss = load_model(model_file, device=accelerator.device,n_hidden=(16,32))
model = accelerator.prepare(model)
        
ss=[]
ys_=[]

#predict
with torch.no_grad():
    model.eval()
    for k, batch in enumerate(loader):
                batch_size = len(batch[0])
                spec,percent= batch[0].float(),batch[1].float()
                s,y_ = model._forward(spec)
                ss.append(s.cpu().numpy())
                ys_.append(y_.cpu().numpy())
    
#save
np.save("./input_dataset/y_test_pred.npy",ys_))
np.save('./input_dataset/latents.npy',ss)
np.save("./input_dataset/seds.npy",seds)
np.save("./input_dataset/percentiles.npy",percentiles)
#np.save("./input_dataset/sfh.npy",ms)


