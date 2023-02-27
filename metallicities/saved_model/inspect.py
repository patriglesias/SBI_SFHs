import numpy as np
#import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


###  Inspect the checkpoint ###
"""
checkpoint = torch.load('./generate_latent_2/checkpoint.pt')
losses=np.array(checkpoint['losses'])
model_loaded=checkpoint['model']
np.savetxt('./generate_latent_2/losses.txt',np.array(losses))
"""

n_latent=16


### Load and visualize losses ###
losses=np.loadtxt('./latent_'+str(n_latent)+'/losses.txt')

"""
#plot losses
epochs=range(len(losses[:,0]))
plt.plot(epochs[:],losses[:,0],label='Training loss')
plt.plot(epochs[:],losses[:,1],label='Validation loss')
plt.xlabel('Epochs')
plt.title('Loss n_latent '+ str(n_latent))
plt.legend()
#plt.ylim(0,5)
#plt.savefig('./generate_latent_2/latent_16/losses.png')
plt.show()
"""


### load and visualize percentiles and latents###

test_set=27000
batch_size=512


percent_pred=np.load('./latent_'+str(n_latent)+'/y_test_pred.npy',allow_pickle=True)
latents=np.load('./latent_'+str(n_latent)+'/latents.npy',allow_pickle=True)
percent=np.load('./latent_'+str(n_latent)+'/y_test.npy',allow_pickle=True)



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
        

for j in tqdm(np.arange(1,11)): #percentiles go from 10% to 90%
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
        
        print('Mean real percentile '+str(j*10)+'% : '+str(np.mean(percent_arr[:,j-1])))
        print('Mean predicted percentile '+str(j*10)+'% : '+str(np.mean(percent_pred_arr[:,j-1])))
        print('Median real percentile '+str(j*10)+'% : '+str(np.median(percent_arr[:,j-1])))
        print('Median predicted percentile '+str(j*10)+'% : '+str(np.median(percent_pred_arr[:,j-1])))
    




for j in range(n_latent):
    plt.hist(latents_arr[:,j])
    plt.title('Latent '+str(j))
    plt.xlabel('Value')
    plt.ylabel('N')
    plt.show()

"""