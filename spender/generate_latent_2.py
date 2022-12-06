#using spectrum encoder and mlp as different models included in 
#encode_percentiles (new class in spender_model)


from pickle import NONE
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
    t,ms,percentiles=generate_weights_from_SFHs(SFR=sfr_linear_exp,mgal=10**10,tau=np.linspace(0.3,5,100),ti=np.linspace(0,5,100),tmin=0,tmax=14,step=0.01,percen=True)
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
          'shuffle': True}


# Datasets 
#percentiles shape(1000, 10)
#seds  shape (1000, 4300)

print('Creating datasets...')

x_train = seds[:int(0.8*len(seds)),:] #seds
y_train = percentiles[:int(0.8*len(seds)),:] #percentiles

x_val = seds[int(0.8*len(seds)):int(0.9*len(seds)),:] #seds
y_val = percentiles[int(0.8*len(seds)):int(0.9*len(seds)),:] #percentiles

x_test = seds[int(0.9*len(seds)):,:] #seds
y_test = percentiles[int(0.9*len(seds)):,:] #percentiles



def train(model, trainloader, validloader, n_epoch=100, n_batch=None, outfile=None, losses=None, verbose=False, lr=3e-4):

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, total_steps=n_epoch)

    accelerator = Accelerator()
    model,  trainloader, validloader, optimizer = accelerator.prepare(model,  trainloader, validloader, optimizer)

    if outfile is None:
        outfile = "./saved_model/generate_latent_2/checkpoint.pt"

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

    for epoch_ in tqdm(range(epoch, n_epoch)):
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


training_mode=False


### TRAINING MODE ###
if training_mode:

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
    model = encoder_percentiles(n_latent=10,n_out=10,n_hidden=(16,16,16),act=None)

    train(model, trainloader, validloader, n_epoch=100, n_batch=128, outfile=None, losses=None, lr=3e-4, verbose=True)

    print('Training has finished')
    print('Model saved')

### TESTING MODE ###
else:
    test_set = Dataset(x_test, y_test)
    print(np.shape(x_test))
    params={'batch_size':len(x_test[:,0]) }
    test_generator = torch.utils.data.DataLoader(test_set,**params) #without minibatches

    print('Calling accelerator...')
    accelerator = Accelerator(mixed_precision='fp16')
    print(accelerator.distributed_type)
    testloader = accelerator.prepare(test_generator)

    print('Loading model...')
    model_file = "./saved_model/generate_latent_2/checkpoint.pt"
    model, loss = load_model(model_file, device=accelerator.device)
    model = accelerator.prepare(model)
    with torch.no_grad():
        model.eval()
        print('Testing starts now...')
        for k, batch in enumerate(testloader):
                batch_size = len(batch[0])
                spec,percent= batch[0].float(),batch[1].float()
                s,y_ = model._forward(spec)
                
    
    
    
    print('Saving latents and predicted percentiles...')
    np.savetxt('./saved_model/generate_latent_2/y_test_pred.txt',y_.cpu())
    np.savetxt('./saved_model/generate_latent_2/latents.txt',s.cpu())
