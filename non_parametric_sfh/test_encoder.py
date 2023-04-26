import numpy as np
import matplotlib.pyplot as plt
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
print('CPU prepared')


#dataset has been generated before, here we just load it
seds=np.load('../../seds_large/non_parametric_sfh/seds_1e5_non_par.npy')
percentiles=np.load('./saved_input/percent_1e5_non_par.npy')


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
n_latent=16


#the dataset has been shuffled and distributed between training, validation and testing before (during training)
ind_sh=np.load('./saved_models/ind_sh_1e5.npy')

seds=seds[ind_sh,:]
percentiles=percentiles[ind_sh,:]

#80% training, 10% validation, 10% test
l=len(seds)
x_test = seds[int(0.9*l):,:] #seds
y_test = percentiles[int(0.9*l):,:] #percentiles

print('Dataset prepared')

print(str(n_latent)+' components selected for the latent vectors')


### TESTING ###
test_set = Dataset(x_test, y_test)
print('Shape of the test set: ',np.shape(x_test))
params={'batch_size': 128 } #no minitbatches or 128
test_generator = torch.utils.data.DataLoader(test_set,**params) #without minibatches

print('Calling accelerator...')
accelerator = Accelerator(mixed_precision='fp16')
print(accelerator.distributed_type)
testloader = accelerator.prepare(test_generator)


print('Loading model...')
model_file = "./saved_models/checkpoint.pt"
model, loss = load_model(model_file, device=accelerator.device,n_hidden=(16,32))
model = accelerator.prepare(model)
        
percentiles=[]
ss=[]
ys_=[]

with torch.no_grad():
    model.eval()
    print('Testing starts now...')
    for k, batch in enumerate(testloader):
                batch_size = len(batch[0])
                spec,percent= batch[0].float(),batch[1].float()
                s,y_ = model._forward(spec)
                percentiles.append(percent.cpu().numpy())
                ss.append(s.cpu().numpy())
                ys_.append(y_.cpu().numpy())
    
    
    
print('Saving latents and predicted percentiles...')
np.save("./saved_models/y_test_pred.npy",ys_)
np.save('./saved_models/latents.npy',ss) 
np.save('./saved_models/y_test.npy', percentiles) 