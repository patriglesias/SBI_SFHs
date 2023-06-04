import numpy as np
import matplotlib.pyplot as plt
import torch
from sbi import utils as Ut
from sbi import inference as Inference
import pickle
from tqdm import tqdm
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
mpl.rcParams["figure.figsize"] = (15,10)

#-------------------------------------------------------

#load latent vectors and percentiles+metallicity
latents_batch=np.load('./saved_models/latents_all.npy',allow_pickle=True)
percentiles=np.load('./saved_input/y.npy',allow_pickle=True)
latents=np.zeros((150000,16))
batch_size=128

#to create a single dataset with all the minibatches
for j in range(len(latents_batch)):
    for i,x in enumerate(latents_batch[j]):
        latents[batch_size*j+i,:]=x

#training with the first 90% elements of the dataset (previously shuffled)

index_sh=np.load('./saved_models/ind_sh.npy')

latents=latents[index_sh,:]
percentiles=percentiles[index_sh,:]

x=latents[:135000,:]
theta=percentiles[:135000,:]

#range for posteriors:

#percentiles
lower_bounds = -2*torch.ones(np.shape(theta[0,:]))
upper_bounds = 16*torch.ones(np.shape(theta[0,:]))

#metallicity
lower_bounds[-1]=-2.5
upper_bounds[-1]=0.6

bounds = Ut.BoxUniform(low=lower_bounds, high=upper_bounds, device='cpu')

#Build and train Normalizing Flows

nhidden = 128 
nblocks = 5
maf_model = Ut.posterior_nn('maf', hidden_features=nhidden, num_transforms=nblocks)
anpe = Inference.SNPE(prior=bounds,
                      density_estimator=maf_model,
                      device='cpu')

#anpe, amortized neural posterior estimation
anpe.append_simulations(
            torch.as_tensor(theta.astype(np.float32)).to('cpu'),
            torch.as_tensor(x.astype(np.float32)).to('cpu'))  

# estimate p(theta|X), the posterior
p_theta_x_est = anpe.train(learning_rate=5e-4)
qphi = anpe.build_posterior(p_theta_x_est)

#Save model

file="./saved_models/my_posterior.pkl"


with open(file, "wb") as handle:
    pickle.dump(qphi, handle)

handle.close()

#Get means and stds for the test sample

n_evaluations=15000
n_samples=1000
index_list=np.arange(135000,150000)

stds=[]
means=[]

for k,j in tqdm(enumerate(index_list)):
    Xobs=latents[j,:]
    posterior_samples= np.array(qphi.sample((n_samples,), x=torch.as_tensor(np.array([Xobs]).astype(np.float32)).to('cpu'), show_progress_bars=False).detach().to('cpu'))
    stds.append(np.std(posterior_samples,axis=0))
    means.append(np.mean(posterior_samples,axis=0))
    
np.save('./saved_models/means.npy',means)
np.save('./saved_models/stds.npy',stds)