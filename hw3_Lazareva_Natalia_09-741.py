import numpy as np
import scipy.fftpack as sf
import matplotlib.pyplot as plt
import scipy.io.wavfile as sw

input = 'voice.wav'
fs, x  = sw.read(input)
time = np.int(fs*20/1000) #20мс
silence = x[0:time]
voice = x[time*1500:time*1500+time]

N_silence = len(silence)
N_voice = len(voice)

#окно Хэннинга
silence = [silence[n]*0.5*(1-np.cos(2*np.pi*n/N_silence)) for n in range(len(silence))]
voice = [voice[n]*0.5*(1-np.cos(2*np.pi*n/N_voice)) for n in range(len(voice))]

f_silence = sf.fft(silence)
f_voice = sf.fft(voice)

plt.figure()
plt.plot(abs(f_silence[:int(len(f_silence)/2)]), 'g')

plt.show()

plt.figure()
plt.plot(abs(f_voice[:int(len(f_voice)/2)]), 'b')
plt.show()


