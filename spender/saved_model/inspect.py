import numpy as np
#import torch
import matplotlib.pyplot as plt

"""
###  Inspect the checkpoint ###
checkpoint = torch.load('./generate_latent_2/checkpoint.pt')
losses=np.array(checkpoint['losses'])
model_loaded=checkpoint['model']
np.savetxt('./generate_latent_2/losses.txt',np.array(losses))
"""


"""
### Load and visualize losses ###
losses=np.loadtxt('./generate_latent_2/losses.txt')
#plot losses
epochs=range(len(losses[:,0]))
plt.plot(epochs,losses[:,0],label='Training loss')
plt.plot(epochs,losses[:,1],label='Validation loss')
plt.xlabel('Epochs')
plt.title('Loss')
plt.legend()
plt.savefig('./generate_latent_2/losses.png')
plt.show()
"""

### load and visualize percentiles and latents###

test_set=100
batch_size=100

percent_pred=np.load('./generate_latent_2/y_test_pred.npy',allow_pickle=True)
latents=np.load('./generate_latent_2/latents.npy',allow_pickle=True)
percent=np.load('./generate_latent_2/y_test.npy',allow_pickle=True)


percent_pred_arr=np.zeros((test_set,10))
latents_arr=np.zeros((test_set,10))
percent_arr=np.zeros((test_set,10))


for j in range(len(percent_pred)):
    for i,x in enumerate(percent_pred[j]):
        percent_pred_arr[batch_size*j+i,:]=x
        
for j in range(len(latents)):
    for i,x in enumerate(latents[j]):
        latents_arr[batch_size*j+i,:]=x
        
for j in range(len(percent)):
    for i,x in enumerate(percent[j]):
        percent_arr[batch_size*j+i,:]=x
        




for j in np.arange(1,10):
    for i in range(test_set): 
        plt.plot(percent_arr[i,j],percent_pred_arr[i,j],'.')#,'k.')
    x=np.arange(np.min(percent_arr[:,j]),np.max(percent_arr[:,j]))
    plt.plot(x,x)
    #plt.title('Percentiles ')
    plt.xlabel('Time percentile '+str(j)+' real (Gyrs)')
    plt.ylabel('Time percentile '+str(j)+' predicted (Gyrs)')
    plt.show()

"""
for j in range(10):
    plt.hist(latents_arr[:,j])
    plt.title('Latent '+str(j))
    plt.xlabel('Value')
    plt.ylabel('N')
    plt.show()
    
"""

