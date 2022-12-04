
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
from accelerate import Accelerator #to use pytorch
from torch.utils.data import DataLoader
from spender import SpectrumEncoder,MLP
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
    wave=npzfile['w']
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
params = {'batch_size': 128,
          'shuffle': True,
          'num_workers': 6}
max_epochs = 100

# Datasets 
#percentiles shape(1000, 10)
#seds  shape (1000, 4300)

print('Creating datasets...')

"""
x_train = seds[:,:] #seds
y_train = percentiles[:,:] #percentiles
"""

x_train = seds[:int(0.8*len(seds)),:] #seds
y_train = percentiles[:int(0.8*len(seds)),:] #percentiles

x_val = seds[int(0.8*len(seds)):int(0.9*len(seds)),:] #seds
y_val = percentiles[int(0.8*len(seds)):int(0.9*len(seds)),:] #percentiles

x_test = seds[int(0.9*len(seds)):,:] #seds
y_test = percentiles[int(0.9*len(seds)):,:] #percentiles



# Generators
training_set = Dataset(x_train, y_train)
training_generator = torch.utils.data.DataLoader(training_set, **params)


validation_set = Dataset(x_val, y_val)
validation_generator = torch.utils.data.DataLoader(validation_set, **params)

"""
print('Calling accelerator...')
dataloader = Accelerator.prepare(training_generator)
"""

print('Training starts now')

print('Initializing both the encoder and the MLP')


#Initialize the encoder

encoder=SpectrumEncoder(instrument=None,n_latent=10,n_hidden=(128, 64, 32),act=None,n_aux=0,dropout=0)

# Initialize the MLP (n_in=n_latent,n_out=n_percentiles)

mlp = MLP(n_in=4300,n_out=10,n_hidden=(16, 16, 16),act=None,dropout=0.5)

# Define the loss function and optimizer
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

#print('Just training dataset')
print('Training and validation datasets')


# Run the training loop
training_loss=[]
latents=[]
y_pred=[]
validation_loss=[]

for epoch in range(0, 15):

    # Print epoch
    print('Starting epoch: ', epoch+1)

    # Set current loss value
    current_t_loss = 0.0
    current_v_loss=  0.0

    # Iterate over the DataLoader for training data
    for i, data in enumerate(training_generator, 0):
        
        # Get and prepare inputs
        x,y = data
        x,y=x.float(),y.float()
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Perform forward pass
        #latent=encoder(x)
        
        outputs = mlp(x)

        #save latents and output percentiles
        y_pred.append(outputs.detach().numpy())
        latents.append(latent.detach().numpy())

        # Compute loss
        loss = loss_function(outputs,y)
        
        # Perform backward pass
        loss.backward()
        
        # Perform optimization
        optimizer.step()
        
        # Print and save statistics
        current_t_loss += loss.item()
    

        if i % 10 == 0:
            print('Training loss after mini-batch: ',current_t_loss)
     
    #save training losses
    training_loss.append(current_t_loss)

    # Iterate over the DataLoader for validation data
    for i, data in enumerate(validation_generator, 0):
        # Get and prepare inputs
        x,y = data
        x,y=x.float(),y.float()
        
        #encode and predict
        #latent=encoder(x)
        outputs = mlp(x)

        # Compute loss
        loss = loss_function(outputs,y)
        
        # Print and save statistics
        current_v_loss += loss.item()
        if i % 10 == 0:
            print('Validation loss after mini-batch: ',current_v_loss)
    #save validation losses
    validation_loss.append(current_v_loss)


    

# Process is complete.
print('Training process has finished.')


#saving losses
np.save('./saved_model/losses.npy',np.array(training_loss))
np.save('./saved_model/val_losses.npy',np.array(validation_loss))
#saving latents and percentiles
try:
    np.save('./saved_model/latents.npy',np.array(latents))
    np.save('./saved_model/y_pred.npy',np.array(y_pred))
except:
    print('error saving latents and percentiles')

plt.plot(range(5),training_loss)
plt.title('Losses')
plt.savefig('./saved_model/loss_curve.png')

#save percentiles (real ones)
np.save('./saved_model/y_train.npy',np.array(y_train))