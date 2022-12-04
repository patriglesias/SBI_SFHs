import numpy as np
import matplotlib.pyplot as plt

"""
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
"""
"""
y_pred=np.load('y_pred.npy',allow_pickle=True)


percentiles=[]
for i in range(len(y_pred)):
    for j in range(len(y_pred[i])):
        percentiles.append(y_pred[i][j])
        
percentiles=np.array(percentiles)


for i in range(len(percentiles[:,0])):
    plt.plot(range(10),percentiles[i,:],'.')
    
plt.title('Predicted Percentiles')
plt.show()

y_train=np.load('y_train.npy',allow_pickle=True)

for i in range(len(y_train[:,0])):
    plt.plot(range(10),y_train[i,:],'.')
    
plt.title('Real Percentiles')
plt.show()
"""

training_loss=np.load('losses.npy',allow_pickle=True)
validation_loss=np.load('val_losses.npy',allow_pickle=True)
#plt.plot(range(5),training_loss,label='Training')
plt.plot(range(5),validation_loss,label='Validation')

plt.title('Losses')
plt.legend()



        