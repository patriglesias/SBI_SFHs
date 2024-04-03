This file explain you how to get along the folder:

1. Simulate: use generate_input.py to load MILES models, create synthetic SFHs and simulate CSP spectra. The simulation output is saved in the saved_input folder.
2. Train the encoder: train the encoder model (architecture in the folder spender) to get useful low-dimensional representations of the spectra with train_encoder.py. The model is saved in the saved_models folder.
3. Test the encoder: test the encoder model with a fraction of the simulated data with test_encoder.py. The results on the test set are saved in the saved_models folder. You can evaluate the performance with the script inspect_encoder_model.py located in the same folder. 
4. Encode all the spectra: run get_latents.py to use the trained encoder model to encode all the simulated spectra. The representations are saved in the saved_models folder.
5. Inspect the low-dimensional representations and create UMAPs with check_latents.ipynb. You require the file line_index.txt to create UMAPs for the line indices. As the computation is expensive we save in the file indices.npy the measured line indices for all the test set.
6. Normalizing Flow: use posterior_estimation.ipynb to define, train and test the Normalizing Flow, saving the model in the folder saved_models.
7. Timer: compute the time required to analyse a spectrum by the encoder and the Normalizing Flow with timer.ipynb.
8. SBC: perform an SBC test with SBC.ipynb, save the output in the folder saved_models.
9. Coverage probability test: do a coverage probability test with coverage_probabilities.ipynb. Not necessary if you previously did the step 8.
