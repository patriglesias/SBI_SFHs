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
from generate_input import sfr_linear_exp,generate_weights_from_SFHs,get_data,get_tbins,interpolate,generate_all_spectrums

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
#torch.cuda.set_device(1)
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
print('CPU prepared')

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

#generate or not the dataset 
generate=False 

n=270000

if generate:
    #generate data:
    print('Generating data...')
    #generate parametrizations
    print('Step 1/4')
    #tau from 0.3 to 5 
    t,ms,percentiles=generate_weights_from_SFHs(SFR=sfr_linear_exp,mgal=10**10,tau=np.logspace(-0.5,0.7,1000),ti=np.linspace(0,5,100),tmin=0,tmax=14,step=0.01,percen=True)
    #load MILES spectra and interpolate
    print('Step 2/4')
    wave,data=get_data(dir_name='../MILES_BASTI_KU_baseFe',strs_1='Mku1.30Zp0.06T',strs_2='_iTp0.00_baseFe.fits')
    tbins=get_tbins(dir_name='../MILES_BASTI_KU_baseFe',strs_1='Mku1.30Zp0.06T',strs_2='_iTp0.00_baseFe.fits')
    print('Step 3/4')
    data_extended=interpolate(tbins,t,data)
    #generate spectra for the parametrized SFHs
    print('Step 4/4')
    wave,seds=generate_all_spectrums(t,ms,wave,data_extended)
    np.save('./saved_input/t_'+str(n)+'.npy',t)
    np.save('./saved_input/percentiles_'+str(n)+'.npy',percentiles)
    np.save('./saved_input/waves_'+str(n)+'.npy',wave)
    np.save('../../seds_'+str(n)+'.npy',seds) #too large file for github
    np.save('../../sfh_'+str(n)+'.npy',ms) #too large file for github


else:
    #load data:
    print('Loading data')
    y=np.load('./saved_input/y.npy')
    wave=np.load('./saved_input/waves_met.npy')
    seds=np.load('../../seds_large/seds_met.npy')
    #ms=np.load('../../sfh.npy') #s



#create a pytorch dataset
print('Creating dataset and calling accelerator')
dataset = Dataset(seds, y)
print('Shape of the dataset: ',np.shape(seds))
params={'batch_size': 512 } 
generator = torch.utils.data.DataLoader(dataset,**params) #with minibatches

#call accelerator
accelerator = Accelerator(mixed_precision='fp16')
loader = accelerator.prepare(generator)

#load model
n_latent=16
print('Loading module trained with latents of ', str(n_latent), ' components')
model_file = "./saved_model/latent_"+str(n_latent)+"/checkpoint.pt"
model, loss = load_model(model_file, device=accelerator.device,n_hidden=(16,32),n_out=10)
model = accelerator.prepare(model)

ss=[]
ys_=[]

#predict
print('Getting latent vectors and predicted percentiles')
with torch.no_grad():
    model.eval()
    for k, batch in enumerate(loader):
                batch_size = len(batch[0])
                spec,percent= batch[0].float(),batch[1].float()
                s,y_ = model._forward(spec)
                ss.append(s.cpu().numpy())
                ys_.append(y_.cpu().numpy())

#save
print('Saving spectra, percentiles, latents and predicted percentiles')
np.save("../SNPE/input_dataset/y_test_pred_"+str(n)+".npy",ys_)
np.save('../SNPE/input_dataset/latents_'+str(n)+'.npy',ss)
np.save("../SNPE/input_dataset/y_"+str(n)+".npy",y)
<<<<<<< HEAD



=======
>>>>>>> b2b77e5231aa8f0579c2797b2df696b88d237331
