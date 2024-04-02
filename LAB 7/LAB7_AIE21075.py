#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import librosa
from hmmlearn import hmm
 
# Load the audio file
file_path = "recordd.wav"  # Assuming the file is in the same directory as your notebook
signal, sr = librosa.load(file_path, sr=None)
 
# Extract STFT features
stft = np.abs(librosa.stft(signal))
 
# Convert STFT features to observation sequence
obs_seq = np.transpose(stft)
 
# Train an HMM model
num_states = 3  # Number of states in the HMM
num_mix = 1     # Number of mixtures in each state
model = hmm.GaussianHMM(n_components=num_states, covariance_type="full", n_iter=1000)
model.fit(obs_seq)
 
# Classify using trained HMM model
predicted_labels = model.predict(obs_seq)
 
# Output the predicted labels
print("Predicted labels:", predicted_labels)


# In[ ]:




