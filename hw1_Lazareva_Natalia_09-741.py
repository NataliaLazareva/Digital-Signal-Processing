#Вычисление отношения сигнал-шум

import numpy as np
import math
import scipy.io.wavfile as w
import matplotlib.pyplot as plt

time = 1
fs = 1000
freq = 10
t = np.linspace(0, time, time*fs)

#генерация сигналов
s1 = np.sin(2*np.pi*freq*t)
s2 = np.random.random_sample((time*fs, ))

b = 16
snr_theory = 6.02*b + 1.7
print('snr_theory = ', snr_theory)

#расчет шума квантования
e1 = np.int16(s1*2**15) - s1*2**15
e2 = np.int16(s2*2**15) - s2*2**15

s1_quant = np.int16(s1*2**15)
s2_quant = np.int16(s2*2**15)

snr_s1 = 10*math.log10(np.var(s1_quant)/np.var(e1))
snr_s2 = 10*math.log10(np.var(s2_quant)/np.var(e2))
print('snr_s1 = ', snr_s1, 'snr_s2 = ', snr_s2)
