# BNN_SFHs
We build a Bayesian Neural Network framework to estimate the SFHs of galaxies from spectra via simulation based inference.

1. We generate a datastet (spectra and mass quantiles) from parametrizations of SFHs (check generate_input.py).
2. We use SPENDER to encode in latent vectors (low dimension representations) the spectra.


1. Generate the input (synthetic) - forward model

- Load MILES SSP spectra with:

A. Solar [M/H] and base [$\alpha$/Fe]
B. Different values of [M/H] and base [$\alpha$/Fe]
C. Different values of [M/H] and [$\alpha$/Fe]

Parameters:
**************
FWMH = 2.51 $\AA$
IMF = KU (1.3)
BaSTI isochrones
$\lambda$ $\in$ [3540.5,7409.6] $\AA$
$\Deta \lambda$ = 0.9 $\AA$
