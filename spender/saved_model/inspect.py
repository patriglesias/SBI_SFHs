import numpy as np
import matplotlib.pyplot as plt

latents=np.load('latents.npy',allow_pickle=True)
print(np.shape(latents))

print(latents[0][0])

print(len(latents[0]))


#latents=np.reshape(latents,(128*40,10))

latent_2=[]
for i in range(len(latents)):
    for j in range(len(latents[i])):
        latent_2.append(latents[i][j])
        
latent_2=np.array(latent_2) #pq 5000? 1000 en c/epoch??

for i in range(len(latent_2[:,0])):
    plt.plot(range(10),latent_2[i,:],'.')
    
plt.title('Components of the latents')
plt.show()
        