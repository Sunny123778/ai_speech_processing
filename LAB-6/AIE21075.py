#!/usr/bin/env python
# coding: utf-8

# In[6]:


#BL.EN.U4AIE21075
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Load the audio file
fs, data = wavfile.read('recordd.wav')

# Function to plot FFT spectrum
def plot_fft(signal, fs, title):
    n = len(signal)
    freq = np.fft.fftfreq(n, d=1/fs)
    magnitude = np.abs(np.fft.fft(signal))
    plt.figure(figsize=(8, 4))
    plt.plot(freq[:n//2], magnitude[:n//2])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.grid(True)
    plt.show()

# FFT for original signal
plot_fft(data, fs, 'Original Signal FFT')


# In[7]:


# A1
vowel_samples = [slice(start, start + fs) for start in range(0, len(data), fs)]
for idx, sample in enumerate(vowel_samples):
    plot_fft(data[sample], fs, f'Vowel Sound {idx + 1}')
    


# In[8]:


# A2
consonant_samples = [slice(start, start + fs) for start in range(fs//2, len(data), fs)]
for idx, sample in enumerate(consonant_samples):
    plot_fft(data[sample], fs, f'Consonant Sound {idx + 1}')


# In[9]:


# A3
silence_samples = [slice(start, start + fs) for start in range(fs//4, len(data), fs)]
for idx, sample in enumerate(silence_samples):
    plot_fft(data[sample], fs, f'Silence/Non-Voiced Sound {idx + 1}')


# In[10]:


# A4
frequencies, times, spectrogram = signal.spectrogram(data, fs)
plt.figure(figsize=(10, 5))
plt.pcolormesh(times, frequencies, np.log(spectrogram))
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.title('Spectrogram')
plt.colorbar(label='Log Power')
plt.show()


# In[ ]:




