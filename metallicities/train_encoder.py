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
from generate_input import sfr_linear_exp,generate_weights_from_SFHs, get_tbins, get_data_met, interpolate_t,generate_all_spectrums



print('Modules prepared')


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
#torch.cuda.set_device(1)
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
print('CPU prepared')

generate=False

if generate:
    #generate data:
    print('Generating data...')
    #generate parametrizations
    print('Step 1/4')
    #tau from 0.3 to 5 
    t,ms,percentiles=generate_weights_from_SFHs(SFR=sfr_linear_exp,mgal=10**10,tau=np.logspace(-0.5,0.7,100),ti=np.linspace(0,5,100),tmin=0,tmax=14,step=0.01,percen=True)
    #load MILES spectra and interpolate
    print('Step 2/4')
    tbins=get_tbins(dir_name='../MILES_BASTI_KU_baseFe',strs_1='Mku1.30Zp0.06T',strs_2='_iTp0.00_baseFe.fits')
    wave,data_met=get_data_met(dir_name='../MILES_BASTI_KU_baseFe',z=np.arange(-2.3,0.4,0.1))
    print('Step 3/4')
    data_extended=interpolate_t(tbins,t,data_met)

    seds=np.zeros((270000,4300))
    metallicity=np.zeros((270000))

    #generate spectra for the parametrized SFHs
    print('Step 4/4')
    z=np.arange(-2.3,0.4,0.1)
    for k,i in enumerate(z):
        print('z: ',i)
        print('iteration: ',k)
        wave,seds[k*10000:(k+1)*10000,:]=generate_all_spectrums(np.arange(0,14+0.01,0.01),ms,wave,data_extended[:,:,k])
        metallicity[k*10000:(k+1)*10000]=i

    np.save('./saved_input/t_met.npy',t)
    np.save('./saved_input/percentiles_met.npy',percentiles)
    np.save('./saved_input/waves_met.npy',wave)
    np.save('../../seds_met.npy',seds) #too large file for github
    np.save('../../sfh_met.npy',ms) #too large file for github
    np.save('./saved_input/met.npy',metallicity)
    
    print(np.shape(metallicity), np.shape(seds),np.shape(percentiles))

else:
    #load data:
    print('Loading data...')
    t = np.load('./saved_input/t_met.npy')
    percentiles=np.load('./saved_input/percentiles_met.npy')
    wave=np.load('./saved_input/waves_met.npy')
    seds=np.load('../../seds_met.npy')
    #ms=np.load('../../sfh.npy')
    metallicity=np.load('./saved_input/met.npy')

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
#percentiles shape(100.000, 9)
#seds  shape (270.000, 4300)
#metalicities (270.000,)


training_mode=False

print('Creating datasets...')

#shuffling (indices are saved for the case of just testing)
if training_mode:
    ind_sh=np.arange(len(seds[:,0]))
    np.random.shuffle(ind_sh)
    np.save('./saved_model/latent_'+str(n_latent)+'/ind_sh.npy',ind_sh)
else:
    ind_sh=np.load('./saved_model/latent_'+str(n_latent)+'/ind_sh.npy')

seds=seds[ind_sh,:]


percentiles_short=np.copy(percentiles)
percentiles=np.zeros((270000,9))

for i in range(27):
    percentiles[10000*i:10000*(i+1),:]=np.copy(percentiles_short)
    
y=np.zeros((270000,10))

for i in range(270000):
    y[i,:9]=percentiles[i,:]
    y[i,-1]=metallicity[i]

#np.save('saved_input/y.npy',y)

y=y[ind_sh,:]

x_train = seds[:int(0.8*len(seds)),:] #seds
y_train = y[:int(0.8*len(seds)),:] #percentiles+metallicity

x_val = seds[int(0.8*len(seds)):int(0.9*len(seds)),:] #seds
y_val = y[int(0.8*len(seds)):int(0.9*len(seds)),:] #percentiles+metallicity

x_test = seds[int(0.9*len(seds)):,:] #seds
y_test = y[int(0.9*len(seds)):,:] #percentiles+metallicity


print(str(n_latent)+' components selected for the latent vectors')

def train(model, trainloader, validloader, n_latent, n_epoch=100, n_batch=None, outfile=None, losses=None, verbose=False, lr=3e-4):

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, total_steps=n_epoch)

    accelerator = Accelerator()
    model,  trainloader, validloader, optimizer = accelerator.prepare(model,  trainloader, validloader, optimizer)

    if outfile is None:
        outfile = "./saved_model/latent_"+str(n_latent)+"/checkpoint.pt"

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
    #(16,32,64)
    #(16,16,16)
    model = encoder_percentiles(n_latent=n_latent,n_out=10,n_hidden=(16,32),act=None,dropout_2=0.0)

    train(model, trainloader, validloader, n_latent,n_epoch=max_epochs, n_batch=batch_size, outfile=None, losses=None, lr=lr, verbose=True)

    print('Training has finished')
    print('Model saved')

    description='n_epochs: %d, batch_size: %d, lr: %.e'%(max_epochs,batch_size,lr)
    print(description)
    f=open('./saved_model/description.txt', "w")
    f.write(description)
    f.close()
  
    checkpoint = torch.load('./saved_model/latent_'+str(n_latent)+'/checkpoint.pt')
    losses=np.array(checkpoint['losses'])
    np.savetxt('./saved_model/latent_'+str(n_latent)+'/losses.txt',np.array(losses))

test_mode=True

if test_mode:
    ### TESTING ###
    test_set = Dataset(x_test, y_test)
    print('Shape of the test set: ',np.shape(x_test))
    params={'batch_size': 512} 
    test_generator = torch.utils.data.DataLoader(test_set,**params) #without minibatches

    print('Calling accelerator...')
    accelerator = Accelerator(mixed_precision='fp16')
    print(accelerator.distributed_type)
    testloader = accelerator.prepare(test_generator)

    if not training_mode:
        print('Loading model...')
        model_file = "./saved_model/latent_"+str(n_latent)+"/checkpoint.pt"
        model, loss = load_model(model_file, device=accelerator.device,n_out=10,n_hidden=(16,32))
        model = accelerator.prepare(model)
            
    ys=[]
    ss=[]
    ys_=[]

    with torch.no_grad():
        model.eval()
        print('Testing starts now...')
        for k, batch in enumerate(testloader):
                    batch_size = len(batch[0])
                    spec,y= batch[0].float(),batch[1].float()
                    s,y_ = model._forward(spec)
                    ys.append(y.cpu().numpy())
                    ss.append(s.cpu().numpy())
                    ys_.append(y_.cpu().numpy())
        
        
        
    print('Saving latents and predicted percentiles...')
    np.save("./saved_model/latent_"+str(n_latent)+"/y_test_pred.npy",ys_)#y_.cpu())
    np.save('./saved_model/latent_'+str(n_latent)+'/latents.npy',ss) #s.cpu())
    np.save('./saved_model/latent_'+str(n_latent)+'/y_test.npy', ys) #,percent.cpu())

    diagnosis=False

    if diagnosis:
        np.save("./saved_model/latent_"+str(n_latent)+"/seds_test.npy",x_test)
