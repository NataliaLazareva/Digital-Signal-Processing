#Генерация водяного знака, наличие его в файле корреляцией
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as sw

# генерация водяного знака
#1 + x^9 + x^11

A = [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]

s0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
s0 = np.reshape(s0, (-1,1))

water = []
d = [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]

for i in range(2**11 - 1):
    s0 = (A @ s0) % 2
    water.append(int(np.dot(s0.T, d) % 2))

water = [(lambda i: -1 if i == 0 else i)(i) for i in water]

#корреляция

sound = 'watermark.wav'
fs, x = sw.read(sound)
cor = np.correlate(x, water, 'valid')
argmax = np.argmax(cor)
print('argmax_p = ', argmax)

plt.figure()
plt.plot(cor, 'purple')
plt.show()