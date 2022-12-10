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

test_set=1000
batch_size=128

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
        


for i in range(test_set): 
    plt.plot(np.arange(10)*10,percent_pred_arr[i,:],'.')
    #plt.plot(np.arange(10)*10,percent_arr[i,:],'.')
plt.title('Percentiles')
plt.xlabel('Percentiles (%)')
plt.xticks(ticks=np.arange(10)*10)
plt.ylabel('Time (Gyrs)')
plt.show()



for i in range(test_set):
    plt.plot(np.arange(10),latents_arr[i,:],'.')
plt.title('Latents')
plt.xlabel('Component')
plt.ylabel('Value')
plt.show()



