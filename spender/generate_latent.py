
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
from accelerate import Accelerator #to use pytorch
from torch.utils.data import DataLoader
from spender import SpectrumEncoder
from generate_input import sfr_linear_exp,generate_weights_from_SFHs,get_data,get_tbins,interpolate,generate_all_spectrums


print('Modules prepared')


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
print('CPU prepared')

generate=False

if generate:    
    #generate data:
    print('Generating data...')
    #generate parametrizations
    t,ms,percentiles=generate_weights_from_SFHs(SFR=sfr_linear_exp,mgal=10**10,tau=np.linspace(0.3,5,100),ti=np.arange(0,5,0.5),tmin=0,tmax=14,step=0.01,percen=True)
    #load MILES spectra and interpolate
    wave,data=get_data(dir_name='../MILES_BASTI_KU_baseFe',strs_1='Mku1.30Zp0.06T',strs_2='_iTp0.00_baseFe.fits')
    tbins=get_tbins(dir_name='../MILES_BASTI_KU_baseFe',strs_1='Mku1.30Zp0.06T',strs_2='_iTp0.00_baseFe.fits')
    data_extended=interpolate(tbins,t,data)
    #generate spectra for the parametrized SFHs
    wave,seds=generate_all_spectrums(t,ms,wave,data_extended)
    np.savez('./saved_input/input.npz',x=t,y=percentiles,w=wave,z=seds)

else:
    #load data:
    print('Loading data...')
    npzfile = np.load('./saved_input/input.npz')
    t=npzfile['x']
    percentiles=npzfile['y']
    w=npzfile['w']
    seds=npzfile['z']


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

print('Creating datasets...')
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

"""
print('Calling accelerator...')
dataloader = Accelerator.prepare(training_generator)
"""

print('Training starts now')

class MLPRegressor(nn.Module):

    def __init__(self, emb_szs, n_cont, out_sz, layers, p=0.5):
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni,nf in emb_szs])
        self.emb_drop = nn.Dropout(p)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        
        layerlist = []
        n_emb = sum((nf for ni,nf in emb_szs))
        n_in = n_emb + n_cont
        
        for i in layers:
            layerlist.append(nn.Linear(n_in,i)) 
            layerlist.append(nn.ReLU(inplace=True))
            layerlist.append(nn.BatchNorm1d(i))
            layerlist.append(nn.Dropout(p))
            n_in = i
        layerlist.append(nn.Linear(layers[-1],out_sz))
            
        self.layers = nn.Sequential(*layerlist)
    
    def forward(self, x_cat, x_cont):
        embeddings = []
        for i,e in enumerate(self.embeds):
            embeddings.append(e(x_cat[:,i]))
        x = torch.cat(embeddings, 1)
        x = self.emb_drop(x)
        
        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x, x_cont], 1)
        x = self.layers(x)
        return x


print('Initializing both the encoder and the MLP')
# Initialize the MLP
mlp = MLPRegressor()
#Initialize the encoder
encoder=SpectrumEncoder()

# Define the loss function and optimizer
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

print('Just training dataset')
# Run the training loop
for epoch in range(0, 100): # 100 epochs at maximum

    # Print epoch
    print(f'Starting epoch {epoch+1}')

    # Set current loss value
    current_t_loss = 0.0
    current_v_loss=  0.0

    # Iterate over the DataLoader for training data
    for i, data in enumerate(training_generator, 0):
        
        # Get and prepare inputs
        seds, percentiles = data
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Perform forward pass
        inputs=encoder(seds)
        outputs = mlp(inputs)
        
        # Compute loss
        loss = loss_function(outputs, percentiles)
        
        # Perform backward pass
        loss.backward()
        
        # Perform optimization
        optimizer.step()
        
        # Print statistics
        current_t_loss += loss.item()
        if i % 10 == 0:
            print('Loss after mini-batch %5d: %.3f' %
                (i + 1, current_loss / 500))
            current_loss = 0.0

# Process is complete.
print('Training process has finished.')






