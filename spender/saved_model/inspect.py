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
"""

### load and visualize percentiles and latents###
percent_pred=np.loadtxt('./generate_latent_2/y_test_pred.txt')
latents=np.loadtxt('./generate_latent_2/latents.txt')
percent=np.loadtxt('./generate_latent_2/y_test.txt')
for i in range(100):
    plt.plot(np.arange(10)*10,percent_pred[i,:],'.')
    plt.plot(np.arange(10)*10,percent[i,:],'.')
plt.title('Percentiles')
plt.xlabel('Percentiles (%)')
plt.xticks(ticks=np.arange(10)*10)
plt.ylabel('Time (Gyrs)')
plt.show()


for i in range(100):
    plt.plot(np.arange(10),latents[i,:],'.')
plt.title('Latents')
plt.xlabel('Component')
plt.ylabel('Value')
plt.show()



