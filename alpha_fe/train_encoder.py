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



#dataset has been generated before, here we just load it
print('Loading data...')
wave=np.load('./saved_input/wave_non_par_alpha.npy')
seds=np.load('../../seds_large/alpha_fe/seds_non_par_alpha_reshaped.npy')
percentiles=np.load('./saved_input/percent_non_par_alpha.npy')
zs = np.load('./saved_input/zs_non_par_alpha.npy')
alpha_fes= np.load('./saved_input/alpha_fes_non_par_alpha.npy')

#Reshape
print('Reshaping...')
#seds=np.reshape(seds,(450000,4300))
percentiles=np.reshape(percentiles,(450000,9))
zs=np.reshape(zs,(450000,))
alpha_fes=np.reshape(alpha_fes,(450000,))

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
max_epochs=50
lr=5e-4
params = {'batch_size': batch_size,
          'shuffle': True}
n_latent=16


# Datasets 
#percentiles+[m/h]+[alpha/fe] shape(450.000, 11)
#seds  shape (450.000, 4300)



print('Creating datasets...')

    
y=np.zeros((len(seds[:,0]),11))

for i in range(len(seds[:,0])):
    y[i,:9]=percentiles[i,:]
    y[i,-2]=zs[i]
    y[i,-1]=alpha_fes[i]



#shuffling (indices are saved for the case of just testing)


#ind_sh=np.arange(len(seds[:,0]))
#np.random.shuffle(ind_sh)
#np.save('./ind_sh_non_par_alpha.npy',ind_sh)

ind_sh=np.load('./saved_models/ind_sh_non_par_alpha.npy')

seds=seds[ind_sh,:]
y=y[ind_sh,:]
zs=zs[ind_sh]
alpha_fes=alpha_fes[ind_sh]


#80% training, 10% validation, 10% test

l=len(seds)

x_train = seds[:int(0.8*l),:] #seds
y_train = y[:int(0.8*l),:] #percentiles

x_val = seds[int(0.8*l):int(0.9*l),:] #seds
y_val = y[int(0.8*l):int(0.9*l),:] #percentiles

x_test = seds[int(0.9*l):,:] #seds
y_test = y[int(0.9*l):,:] #percentiles


print(str(n_latent)+' components selected for the latent vectors')

def train(model, trainloader, validloader, n_latent, n_epoch=100, n_batch=None, outfile=None, losses=None, verbose=False, lr=3e-4):

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, total_steps=n_epoch)

    accelerator = Accelerator()
    model,  trainloader, validloader, optimizer = accelerator.prepare(model,  trainloader, validloader, optimizer)

    if outfile is None:
        outfile = "./saved_models/checkpoint.pt"

    epoch = 0
    if losses is None:
        losses = []
    else:
        try:
            epoch = len(losses)
            n_epoch += epoch
            if verbose:
                train_loss, valid_loss = losses[-1]
                print(f'====> Epoch: {epoch-1} TRAINING Loss: {train_loss:.3e}  VALIDATION Loss: {valid_loss:.3e}')
        except: # OK if losses are empty
            pass

    for epoch_ in trange(epoch, n_epoch):
        model.train()
        train_loss = 0.
        n_sample = 0
        
        for k, batch in enumerate(trainloader):
            batch_size = len(batch[0])
            spec,percent = batch[0].float(),batch[1].float()
            loss = model.loss(percent)
            accelerator.backward(loss)
            train_loss += loss.item()
            n_sample += batch_size
            optimizer.step()
            optimizer.zero_grad()

            # stop after n_batch
            if n_batch is not None and k == n_batch - 1:
                break
        train_loss /= n_sample

        with torch.no_grad():
            model.eval()
            valid_loss = 0.
            n_sample = 0
            for k, batch in enumerate(validloader):
                batch_size = len(batch[0])
                spec,percent= batch[0].float(),batch[1].float()
                loss = model.loss(percent)
                valid_loss += loss.item()
                n_sample += batch_size
                # stop after n_batch
                if n_batch is not None and k == n_batch - 1:
                    break
            valid_loss /= n_sample

        scheduler.step()
        losses.append((train_loss, valid_loss))

        if verbose:
            print(f'====> Epoch: {epoch_} TRAINING Loss: {train_loss:.3e}  VALIDATION Loss: {valid_loss:.3e}')

        # checkpoints
        if epoch_ % 5 == 0 or epoch_ == n_epoch - 1:
            unwrapped_model = accelerator.unwrap_model(model)
            accelerator.save({
                "model": unwrapped_model.state_dict(),
                "losses": losses,
            }, outfile)


### TRAINING MODE ###


# Generators
training_set = Dataset(x_train, y_train)
training_generator = torch.utils.data.DataLoader(training_set, **params)


validation_set = Dataset(x_val, y_val)
validation_generator = torch.utils.data.DataLoader(validation_set, **params)


print('Calling accelerator...')
accelerator = Accelerator(mixed_precision='fp16')
print(accelerator.distributed_type)
trainloader = accelerator.prepare(training_generator)
validloader= accelerator.prepare(validation_generator)

print('Training starts now...')

# define and train the model
print('Model defined')
#(16,32,64)
#(16,16,16)
model = encoder_percentiles(n_latent=n_latent,n_out=11,n_hidden=(16,32),act=None,dropout_2=0.0)

train(model, trainloader, validloader, n_latent,n_epoch=max_epochs, n_batch=batch_size, outfile=None, losses=None, lr=lr, verbose=True)

print('Training has finished')
print('Model saved')

description='n_epochs: %d, batch_size: %d, lr: %.e'%(max_epochs,batch_size,lr)
print(description)
f=open('./saved_models/description.txt', "w")
f.write(description)
f.close()
  
checkpoint = torch.load('./saved_models/checkpoint.pt')
losses=np.array(checkpoint['losses'])
np.savetxt('./saved_models/losses.txt',np.array(losses))




