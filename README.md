# BNN_SFHs
We build a Bayesian Neural Network framework to estimate the SFHs of galaxies from spectra via simulation based inference.

## 1. Generate the input (synthetic) - forward model

### 1.1. Load MILES SSP spectra with:

- Solar [M/H] and base [$\alpha$/Fe]
- Different values of [M/H] and base [$\alpha$/Fe]
- Different values of [M/H] and [$\alpha$/Fe]

Parameters:
**************
- FWMH = 2.51 $\AA$
- IMF = KU (1.3)
- BaSTI isochrones
- $\lambda$ $\in$ [3540.5,7409.6] $\AA$
- $\Delta \lambda$ = 0.9 $\AA$
- t $\in$ [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,  0.10,   0.15,  0.20,   0.25,  0.30,
  0.35,  0.40,   0.45  0.50,   0.60,   0.70,   0.80,   0.90,   1.00,    1.25,  1.50,   1.75,
  2.00,    2.25,  2.50,   2.75,  3.00,    3.25,  3.50,   3.75,  4.00,    4.50,   5.00,    5.50,
  6.00,    6.50,   7.00,    7.50,   8.00,    8.50,   9.00,    9.50,  10.00,   10.50,  11.00,   11.50,
 12.00,   12.50,  13.00,   13.50,  14.00 ] Gyr
- [M/H] $\in$ [-2.27, -1.79, -1.49, -1.26, -0.96, -0.66, -0.35, -0.25,  0.06,  0.15,  0.26,  0.40 ]
- [$alpha$/Fe] $\in$ [0.00,0.40]

### 1.2 Generate SFHs

We fix $M_{*}=10^{10} M_{\odot}$ and $z=0.0$

- Linear-exponential parametrization: t $\in$ [0.00,14.00] Gyr, $\Delta$ t = 0.01 Gyr, $t_0 \in$ [0.00,5.00] Gyr, 
$\Delta t_0$ = 0.05 Gyr, $\tau \in$ [$10^{-0.5}$,$10^{0.7}$] Gyr, $\Delta \tau$ = $10^{0.012}$ Gyr. In total 10.000 different SFHs for each pair [M/H], [$alpha$/Fe].


$$
S F R\left(t, t_0, \tau\right)=\Theta\left(t-t_0\right)\left(\left(t-t_0\right) / \tau\right) e^{-\left(t-t_0\right) / \tau}
$$

- Non parametric SFHs: generated by the module dense_basis (GP-SFH) [Iyer16], include a combination of SAMs, hidrodynamical simulations and stochastic SFHs. In total 10.000 different SFHs for each pair [M/H], [$alpha$/Fe].

Once we get the SFHs we integrate them to get 9 percentiles: the time at which the stellar mass becomes 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80% and 90% of the total mass. It seems preferable to use them as input of the model as they are more robust quantities.

### 1.3 Get the spectra for each SFH

First, we interpolate the MILES library to get values of [M/H] in the same range but each 0.2 units, and three different values for [$alpha$/Fe]: 0.00, 0.20 and 0.40. We also get SSP spectra each 0.01 Gyr for the parametric SFHs, or, for the non-parametric SFHs, in all the bins given by the GP-SFH module, with t $\in$ [0.00,13.47] Gyr and $\Delta$t=0.013 Gyr.

For each SFH, given the values of [M/H] and [$alpha$/Fe], we combine all the MILES interpolated spectra with these values (corresponding to different ages) as:

$$ 
\text{Galaxy Spectra}(M_{tot},[M/H],[\alpha / Fe])=\sum_{t_i} \frac{M(t_i)}{M_{tot}} \cdot spectra(t_i,[M/H],[\alpha/Fe])
$$

And, as indicated above, we fix $M_{tot}=10^10 M_{\odot}$.

For example, for the non-parametric sample,  we get 10.000 SFHs for each pair [M/H], [$alpha$/Fe], so: 10.000 SFHs x 15 bins of metallicity x 3 bins of $\alpha$ enhacement = 450.000 samples.

## 2. Encode spectra (encoding architecture by SPENDER [Melchior22])

We want to encode 4300-components vectors (the spectra) into 16-components latent vectors to extract the most important features, so it is easier for the bayesian model to learn (skip unuseful information to accelerate training) and reconstruct SFHs, as well as metallicity and $\alpha$ enhacement.

The network consists on a 3-layers CNN (moving to wider kernels and including max-pooling) + an attention module (dot-product) + extra-CNN to optimize encoding for our goal (obtain 9 percentiles and a value for [M/H] and [$\alpha/Fe], incorporating a convenient loss function).

## 3. Interpretate latent vectors

As the input of the bayesian network, we need to be sure the latent vectors contain relevant information. However, due to its high dimensionality, they are difficult to  interpretate. We carry out the following steps:

- Study of the distribution of the 16 components.
- Study of the correlation of the 16 components between them.
- Study of the correlation of the 16 components with the spectral regions (wavelengths).
- UMAPs: another encoding of the 16 components into a plane (2 components) using colormaps according to the flux in specific spectral regions ($H_{\alpha}$, $H_{\beta}$...), to the percentiles or to the metallicity / $\alpha$ enhacement.


## 4. Use ANPE to obtain posteriors - backward model

We employ Bayesian inference (Amortized Neural Posterior Estimation) to estimate the posterior probability distribution for each of the 9 percentiles, as well as for the metallicity and $\alpha$ enhacement. We use 90% of the generated samples (x= latent vectors, y=(percentiles,[M/H],[$\alpha$/Fe])) to train the network. 

## 5. Evaluate the performance with the test set 

We use the remaining 10% of the samples to evaluate the performance of the model, by getting from their latent vectors probability distributions for the values of each of the 9 percentiles + [M/H] + [$\alpha$/Fe], and comparing them with the actual true values.

The general performance is obtained by plotting the mean value of each posterior vs the ground truth, while the uncertainty prediction is checked studying the coverage probabilities (our model is too conservative or too confident?)

## 6. Timing

Once the model (encoder + ANPE) is trained, we estimate how much time it takes to predict the 9 percentiles + [M/H] + [$\alpha$/Fe] given one spectra. We can compare it with some traditional methods like MCMC.

## 7. Recover spectra

When working with real galaxy observations, we will not be able to compare the predicted values with the ground truth. To evaluate the performance of the model in these situations we can try to recover the spectra by repeating the forward model with the predicted values (mean of the posteriors), taking into acount that $\frac{d \text{percentiles}}{dt}$ = SFH, and check how good is the reconstruction (residuals between real and reconstructed spectra).

## 8. Test with observations

We use the model trained with synthetic spectra and non-parametric SFHs to predict the 9 percentiles + [M/H] + [$\alpha$/Fe] for observed early-type galaxies from their spectra.