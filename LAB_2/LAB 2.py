#!/usr/bin/env python
# coding: utf-8

# In[17]:


##Q1
import librosa
import matplotlib.pyplot as plt
y, rs = librosa.load('recordd.wav')
plt.figure(figsize=(15, 5))
librosa.display.waveshow(y, sr=rs,color='brown')
plt.title('The Actual One')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.show()
ipd.Audio(y, rate=rs)


# In[9]:


import IPython.display as ipd
import numpy as np
import librosa
import matplotlib.pyplot as plt
y, rs = librosa.load('recordd.wav')
derivative_1 = np.diff(y)
derivative_1 /= np.max(np.abs(derivative_1))
plt.figure(figsize=(15, 5))
librosa.display.waveshow(derivative_1, sr=rs,color='blue')
plt.title('first derivative signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.show()
print("First Derivative Signal:")
ipd.Audio(derivative_1, rate=rs)


# In[19]:


##Q2
zero_crossing = np.where(np.diff(np.sign(derivative_1)))[0]
diff = np.diff(zero_crossing)
threshold = 1000
speech_regions = diff[diff > threshold]
silence_regions = diff[diff <= threshold]

avg_length_speech = np.mean(speech_regions)
avg_length_silence = np.mean(silence_regions)

print("Average length between consecutive zero crossings in speech regions:", avg_length_speech)
print("Average length between consecutive zero crossings in silence regions:", avg_length_silence)

plt.figure(figsize=(10, 5))
plt.plot(diff, label='All regions',color = 'yellow')
plt.plot(np.arange(len(speech_regions)), speech_regions, 'ro', label='Speech regions',color = 'red')
plt.plot(np.arange(len(speech_regions), len(speech_regions) + len(silence_regions)), silence_regions, 'bo', label='Silence regions',color = 'blue')
plt.title('Pattern of Zero Crossings')
plt.xlabel('Zero Crossing Difference')
plt.ylabel('Difference between Consecutive Zero Crossings')
plt.legend()
plt.show()

print("Pattern of Zero Crossings:")
print("All regions:", diff)
print("Speech regions:", speech_regions)
print("Silence regions:", silence_regions)


# In[29]:


##Q3
word_files_teammate = ['apple.mp3', 'ball.mp3', 'cat.mp3', 'dog.mp3', 'elephant.mp3']
word_files_mine = ['apple.wav', 'ball.wav', 'cat.wav', 'dog.wav', 'elephant.wav']
words = ['apple', 'ball', 'cat', 'dog', 'elephant']
word_lengths_mine = []
word_lengths_teammate = []

for word_file in word_files_mine:
    signal, sr = librosa.load(word_file, sr=None)
    length_seconds = len(signal) / sr
    word_lengths_mine.append(length_seconds)

for word_file in word_files_teammate:
    signal, sr = librosa.load(word_file, sr=None)
    length_seconds = len(signal) / sr
    word_lengths_teammate.append(length_seconds)

print("Lengths of the spoken words MINE:", word_lengths_mine)
print("Lengths of the spoken words Teammate:", word_lengths_teammate)

bar_width = 0.35
index = np.arange(len(words))
plt.figure(figsize=(12, 6))
plt.bar(index - bar_width/2, word_lengths_mine, bar_width, label='My Words', color='black')
plt.bar(index + bar_width/2, word_lengths_teammate, bar_width, label="Teammate's Words", color='blue')
plt.xlabel('Words')
plt.ylabel('Length (seconds)')
plt.title('Comparison of Spoken Words Length')
plt.xticks(index, words)
plt.legend()

plt.show()


# In[28]:


##Q4

statement, sr1 = librosa.load('statement.wav')
question, sr2 = librosa.load('question.wav')
plt.figure(figsize=(15, 5))
librosa.display.waveshow(statement, sr=sr,color='red')
plt.title('STATEMENT')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.show()

question, sr = librosa.load('question.wav')
plt.figure(figsize=(15, 5))
librosa.display.waveshow(statement, sr=sr,color='yellow')
plt.title('QUESTION')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:




