This file explains how to use the folder:

1. Simulate: use generate_input.py to load MILES models, generate synthetic SFHs and simulate CSP spectra. The simulation output is stored in the saved_input folder.
2. Train the encoder: train the encoder model (architecture in the encoder folder) to obtain useful low-dimensional representations of the spectra using train_encoder.py. The model will be saved in the saved_models folder.
3. Test the encoder: test the encoder model with a fraction of the simulated data using test_encoder.py. The results of the test set are saved in the saved_models folder. You can evaluate the performance with the script inspect_encoder_model.py in the same folder. 
4. Encoding all spectra: run get_latents.py to use the trained encoder model to encode all simulated spectra. The representations are stored in the saved_models folder.
5. Check the low-dimensional representations and generate UMAPs using check_latents.ipynb. You will need the file line_index.txt to generate UMAPs for the line indices. As the computation is expensive, we store the measured line indices for the whole test set in the file indices.npy.
6. Normalising flow: use posterior_estimation.ipynb to define, train and test the normalising flow and save the model in the saved_models folder.
7. Timer: use timer.ipynb to calculate the time taken by the encoder and the normalising flow to analyse a spectrum.
8. SBC: perform an SBC test with SBC.ipynb, saving the output in the saved_models folder.
9. Coverage probability test: run a coverage probability test with coverage_probabilities.ipynb. Not necessary if you have already done step 8.
