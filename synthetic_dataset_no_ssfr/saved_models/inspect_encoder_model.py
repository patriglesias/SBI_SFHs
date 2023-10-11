import numpy as np
#import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

import matplotlib as mpl

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['xtick.labelsize'] = 'medium'
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.labelsize'] = 'medium'
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['legend.frameon'] = True
mpl.rcParams['font.size'] = 30
mpl.rcParams["figure.figsize"] = (15,7)


###  Inspect the checkpoint ###
"""
checkpoint = torch.load('./checkpoint.pt')
losses=np.array(checkpoint['losses'])
model_loaded=checkpoint['model']
np.savetxt('./losses.txt',np.array(losses))
"""

n_latent=16


### Load and visualize losses ###
losses=np.loadtxt('./losses.txt')


#plot losses
epochs=range(len(losses[:,0]))
plt.plot(epochs[:],losses[:,0],label='Training loss')
plt.plot(epochs[:],losses[:,1],label='Validation loss')
plt.yscale('log')
plt.yticks([1e-1,1e-2,1e-3])
plt.xlim(-1,251)
plt.xlabel('Epochs')
plt.title('Encoder loss')
plt.text(92,1.25e-2,'Log(cosh(y\'- y))')
plt.tight_layout()
plt.legend()
#plt.ylim(0,5)
#plt.savefig('./losses.jpg')
plt.show()

### load and visualize percentiles and latents###

test_set=150000 #here we had a problem because with saved
#with the same name all the predictions (not only test set)
batch_size=128


ind_sh=np.load('./ind_sh.npy',allow_pickle=True)
print(np.shape(ind_sh))
percent_pred=np.load('./y_test_pred.npy',allow_pickle=True)
latents=np.load('./latents.npy',allow_pickle=True)
percent=np.load('./y_test.npy',allow_pickle=True)




percent_pred_arr=np.zeros((test_set,10))
latents_arr=np.zeros((test_set,n_latent))
percent_arr=np.zeros((test_set,10))


for j in tqdm(range(len(percent_pred))):
    for i,x in enumerate(percent_pred[j]):
        percent_pred_arr[batch_size*j+i,:]=x
        
for j in tqdm(range(len(latents))):
    for i,x in enumerate(latents[j]):
        latents_arr[batch_size*j+i,:]=x
        
for j in tqdm(range(len(percent))):
    for i,x in enumerate(percent[j]):
        percent_arr[batch_size*j+i,:]=x
        

for j in [1,2]: #tqdm(np.arange(1,10)): #percentiles go from 10% to 90%
    print('j')    
    for i in range(test_set)[::10]: 
        plt.plot(percent_arr[i,j-1],percent_pred_arr[i,j-1],'.')#,'k.')
    print('x')
    x=np.arange(np.min(percent_arr[:,j-1]),np.max(percent_arr[:,j-1]))
    plt.plot(x,x)
    #plt.title('Percentiles ')
    plt.xlabel('Time percentile '+str(j*10)+' real (Gyrs)')
    plt.ylabel('Time percentile '+str(j*10)+' predicted (Gyrs)')
    plt.show()

"""
#histogram for percentiles and means
for j in np.arange(1,11): #percentiles go from 10% to 90%
        plt.hist(percent_arr[:,j-1],color='r',label='Real')
        plt.hist(percent_pred_arr[:,j-1],color='b',label='Predicted')#,'k.')
        plt.ylabel('N')
        plt.xlabel('Time percentile '+str(j*10)+' (Gyrs)')
        plt.legend()
        plt.show()
        

    """
