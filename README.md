# BNN_SFHs
We build a Bayesian framework to estimate the SFHs of galaxies from spectra via simulation-based inference.

## 1. Generate the input (synthetic) - forward model

### 1.1. Load MILES SSP spectra

We use different values of $[M/H]$ and base $[\alpha/Fe]$


Parameters:
**************
- FWHm $=$ $2.51$ Å
- IMF $=$ KU ($1.3$)
- BaSTI isochrones
- $\lambda$ $\in$ $[3540.5,7409.6]$ Å
- $\Delta \lambda = 0.9$ Å
- $t \in$ $[0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,  0.10,   0.15,  0.20,   0.25,  0.30,
  0.35,  0.40,   0.45  0.50,   0.60,   0.70,   0.80,   0.90,   1.00,    1.25,  1.50,   1.75,
  2.00,    2.25,  2.50,   2.75,  3.00,    3.25,  3.50,   3.75,  4.00,    4.50,   5.00,    5.50,
  6.00,    6.50,   7.00,    7.50,   8.00,    8.50,   9.00,    9.50,  10.00,   10.50,  11.00,   11.50,
  12.00,   12.50,  13.00,   13.50,  14.00] $ Gyr
- $[M/H]$ $\in$ $[-2.27, -1.79, -1.49, -1.26, -0.96, -0.66, -0.35, -0.25,  0.06,  0.15,  0.26,  0.40]$
- $[\alpha/Fe]$ solar-scaled

![Different Ages](https://github.com/patriglesias/BNN_SFHs/blob/2ac3d3629b2c8993c6905aa41c537c344b4d33ec/synthetic_dataset/plots/spectra_different_ages.pdf)

  
![Different Metallicities](https://github.com/patriglesias/BNN_SFHs/blob/2ac3d3629b2c8993c6905aa41c537c344b4d33ec/synthetic_dataset/plots/spectra_different_metallicities.pdf)

### 1.2 Generate SFHs


 Non-parametric SFHs generated by the module dense_basis (GP-SFH) [Iyer16], including a combination of SAMs, hydrodynamical simulations and stochastic SFHs. In total $10.000$ different SFHs for each $[M/H]$. We fix  $z=0.0$ and by the moment we avoid selecting a prior on the mass.

![Non-parametric SFHs](https://github.com/patriglesias/BNN_SFHs/blob/2ac3d3629b2c8993c6905aa41c537c344b4d33ec/synthetic_dataset/plots/non_param_sfhs.pdf)
![Percentiles non-parametric SFHs](https://github.com/patriglesias/BNN_SFHs/blob/2ac3d3629b2c8993c6905aa41c537c344b4d33ec/synthetic_dataset/plots/non_param_percentiles.pdf)


Once we get the SFHs we integrate them to get $9$ percentiles: the time at which the stellar mass becomes $10$%, $20$%, $30$%, $40$%, $50$%, $60$%, $70$%, $80$% and $90$% of the total mass. It seems preferable to use them as input of the model as they are more robust quantities.

### 1.3 Get the spectra for each SFH

First, we interpolate the MILES library to get values of $[M/H]$ in the same range but equally spaced. We also get SSP spectra in all the bins given by the GP-SFH module, with $t \in$ $[0.00,13.47]$ Gyr and $\Delta t=0.013$ Gyr.

For each SFH, given the value of $[M/H]$, we combine all the MILES interpolated spectra with that value (corresponding to different ages) as:

$F_{\text{gal}} \left( \lambda,M_{\text{tot}},[M/H],[\alpha / Fe]_{\text{Base}} \right) = \sum_{t_i} \frac{M(t_i)}{M_{\text{tot}}} \cdot F_{\text{SSP}} \left( \lambda,t_i,[M/H],[\alpha/Fe]_{\text{Base}}  \right)$

And we normalize each spectrum by its median.
  
We get $10.000$ SFHs for each $[M/H]$, so: $10.000$ SFHs x $15$ bins of metallicity $= 150.000$ samples.

## 2. Encode spectra (encoding architecture by SPENDER [Melchior22])

We want to encode $4300$-components vectors (the spectra) into $16$-components latent vectors to extract the most important features, so it is easier for the Bayesian model to learn (skip useless information to accelerate training) and reconstruct the SFHs, as well as the metallicity.

The network consists of a $3$-layer CNN (moving to wider kernels and including max-pooling) + an attention module (dot-product) + extra-CNN to optimize encoding for our goal (obtain $9$ percentiles and a value for $[M/H]$, incorporating a convenient loss function).

![Encoder diagram]()

## 3. Interpretation of the  latent vectors

As the input of the Bayesian network, we need to be sure the latent vectors contain relevant information. However, due to their high dimensionality, they are challenging to  interpret. We carry out the following steps:


- Study of the correlation of the $16$ components between them.
- Study of the correlation of the $16$ components with the spectral regions (wavelengths)
- UMAPs: another encoding of the $16$ components into a plane ($2$ components) using colormaps according to the percentiles, the metallicity or line indices.

![Correlation latents]()

![Correlation latents with spectral regions]()
![Correlation latents with spectral regions]()
![Correlation latents with spectral regions]()



![UMAP percentile 10%]()
![UMAP percentile 90%]()
![UMAP metallicity]()
![UMAP Hbeta]()
![UMAP Mgb177]()

## 4. Use Normalizing Flows to obtain posteriors - backward model

We employ Bayesian inference (Amortized Neural Posterior Estimation) to estimate the posterior probability distribution for each of the $9$ percentiles, as well as for the metallicity. We use $90$% of the generated samples ( $x =$ {latent vectors}, $y =$ {percentiles, $[M/H]$}) to train the network, a Masked Autoregressive Flow. 

![BNN diagram](https://github.com/patriglesias/BNN_SFHs/blob/2fd75d6bc874adf295b364da9e416e78cf536d25/img_readme/SNPE_SBI.png)


## 5. Evaluate the performance with the test set 

We use the remaining $10$% of the samples to evaluate the performance of the model, by getting from their latent vectors probability distributions for the values of each of the $9$ percentiles + $[M/H]$ and comparing them with the actual true values.

The general performance is obtained by plotting the mean value of each posterior vs the ground truth, while the uncertainty prediction is checked by studying the coverage probabilities. We also do a Corner plot.

![Examples percentile prediction]()
![Examples percentile prediction]()
![Examples percentile prediction]()
![Examples percentile prediction]()


![Mean posterior vs ground truth]()
![Mean posterior vs ground truth]()
![Mean posterior vs ground truth]()
![Mean posterior vs ground truth]()

![Corner plot]()
![Coverage probabilities]()


## 6. Timing

Once the model (encoder + ANPE) is trained, we estimate how much time it takes to predict the $9$ percentiles + $[M/H]$ given one spectrum. We can compare it with some traditional methods like MCMC.



## 7. Test with observations

We use the model trained with synthetic spectra and non-parametric SFHs to predict the $9$ percentiles + $[M/H]$ + $[\alpha/Fe]$ for observed early-type galaxies from their spectra.

First, we convolve all the observations to emulate the maximum velocity dispersion of the dataset: $\sigma=300$ Km/s (convolve with $\sigma^2=\sigma_{300}^2+\sigma_{\text{sdss}}^2 - (\sigma_{v_{i}}^2+\sigma_{\text{sdss}}^2)=\sigma_{300}^2-\sigma_{v_{i}}^2$). Then, we limit the wavelength range to $[4023,5500]$ Å. On the other hand, we process the MILES spectra in order to simulate the conditions of the observations: we convolve with a kernel of size $\sigma^2=\sigma_{300}^2+\sigma_{\text{sdss}}^2 -\sigma_{\text{MILES}}$, we interpolate to get $\Delta \lambda = 1$ Å as in the observations, and we clip the spectra to $[4023,5500]$ Å too. Once we have all with the same resolution and wavelength range, we repeat the training with these MILES spectra, testing again its performance, and checking how the resolution and the wavelength range affect it. Eventually, we get the posteriors for the observations and analyze them, obtaining an SFH, a metallicity and an $[\alpha/Fe]$ value for each observed galaxy.


![Predictions for observations]()
![Predictions for observations]()
![Predictions for observations]()


![All predictions for observations]()



## 8. Recover spectra

When working with real galaxy observations, we are not able to compare the predicted values with the ground truth. To evaluate the performance of the model in these situations we can try to recover the spectra by repeating the forward model with the predicted values (mean of the posteriors), taking into account that $\frac{\partial  \text{percentiles}}{\partial t} =$ SFH, and check how good is the reconstruction (residuals between real and reconstructed spectra).

![Examples recover spectra]()
![Examples recover spectra from predicted]()
![Examples recover spectra from predicted]()
