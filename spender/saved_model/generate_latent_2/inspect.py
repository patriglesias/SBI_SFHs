import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


checkpoint = torch.load('checkpoint.pt')

losses=checkpoint['losses']
model_loaded=checkpoint['model']


print(np.shape(losses))
