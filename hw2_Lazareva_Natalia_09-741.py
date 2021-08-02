#Реализовать sinс функцию h(t,T).
#Показать на примере синуса восстановление сигнала с помощью sinc функции.
import numpy as np
import matplotlib.pyplot as plt

time = 1
fs = 1000
freq = 10
t = np.linspace(0, time, time*fs)
s = np.sin(2*np.pi*freq*t)

#по теореме Котельникова
fs1 = 25
T1 = 1/fs1
t1 = np.linspace(0, time, time*fs1)
s1 = np.sin(2*np.pi*fs1*t1)
s1_interp = np.zeros(len(t))

#не по теореме Котельникова
fs2 = 9
T2 = 1/fs2
t2 = np.linspace(0, time, time*fs2)
s2 = np.sin(2*np.pi*fs2*t2)
s2_interp = np.zeros(len(t))

for n in range(len(s1)):
    s1_interp += T1 * s1[n] * np.sin((t - t1[n]*T1)*np.pi*fs1) / ((t-t1[n])*np.pi)

for n in range(len(s2)):
    s2_interp += T2 * s2[n] * np.sin((t - t2[n]*T2)*np.pi*fs2) / ((t-t2[n])*np.pi)

plt.figure()
plt.plot(t, s)
plt.plot(t, s1_interp)
plt.show()

plt.figure()
plt.plot(t, s)
plt.plot(t, s2_interp)
plt.show()