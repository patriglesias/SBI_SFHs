import numpy as np
import torch
import matplotlib.pyplot as plt

"""
checkpoint = torch.load('./generate_latent_2/checkpoint.pt')
losses=np.array(checkpoint['losses'])
model_loaded=checkpoint['model']
np.savetxt('./generate_latent_2/losses.txt',np.array(losses))
"""

losses=np.loadtxt('./generate_latent_2/losses.txt')

#plot losses
epochs=range(len(losses[:,0]))
plt.plot(epochs,losses[:,0],label='Training loss')
plt.plot(epochs,losses[:,1],label='Validation loss')
plt.xlabel('Epochs')
plt.title('Loss')
plt.savefig('./generate_latent_2/losses.png')
