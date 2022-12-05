import numpy as np
import torch
#import torch.nn as nn
#import torch.optim as optim
from accelerate import Accelerator #to use pytorch

checkpoint = torch.load('checkpoint.pt')

losses=np.array(checkpoint['losses'])
model_loaded=checkpoint['model']


print(np.shape(losses))
np.savetxt('./generate_latent_2/losses.txt',np.array(losses))
