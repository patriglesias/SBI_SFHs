# ## Load model and get a dataset of spectra - latents - percentiles #

print('Loading modules')

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
from accelerate import Accelerator #to use pytorch
from torch.utils.data import DataLoader
from spender import SpectrumEncoder,MLP,encoder_percentiles,load_model

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
#torch.cuda.set_device(1)
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
print(device,' prepared')

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



n=150

#load data
print('Loading data')

percentiles=np.load('./saved_input/percent_1e5_non_par.npy')
seds=np.load('../../seds_large/seds_1e5_non_par.npy')

#create a pytorch dataset
print('Creating dataset and calling accelerator')
dataset = Dataset(seds, percentiles)
print('Shape of the dataset: ',np.shape(seds))
params={'batch_size': 128 } 
generator = torch.utils.data.DataLoader(dataset,**params) #with minibatches

#call accelerator
accelerator = Accelerator(mixed_precision='fp16')
loader = accelerator.prepare(generator)

#load model
n_latent=16
print('Loading module trained with latents of ', str(n_latent), ' components')
model_file = "./saved_models/checkpoint.pt"
model, loss = load_model(model_file, device=accelerator.device,n_hidden=(16,32))
model = accelerator.prepare(model)

ss=[]
ys_=[]

#predict
print('Getting latent vectors and predicted percentiles')
with torch.no_grad():
    model.eval()
    for k, batch in tqdm(enumerate(loader)):
                batch_size = len(batch[0])
                spec,percent= batch[0].float(),batch[1].float()
                s,y_ = model._forward(spec)
                ss.append(s.cpu().numpy())
                ys_.append(y_.cpu().numpy())

#save
print('Saving spectra, percentiles, latents and predicted percentiles')
np.save("./saved_models/y_test_pred_"+str(n)+".npy",ys_)
np.save("./saved_models/latents_"+str(n)+".npy",ss)
np.save("./saved_models/percentiles_"+str(n)+".npy",percentiles)

