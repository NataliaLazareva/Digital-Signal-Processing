import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft, rfftfreq
import scipy.signal as sgn

def fir_filter(order, new_freqs_array):
  #полосопропускающий фильтр
  gain = [0, 1, 1, 0]
  freq = [0, new_freqs_array[2], new_freqs_array[4], 1]
  b = sgn.firwin2(order, freq, gain)
  w, h = sgn.freqz(b)
  return b, w/np.pi, abs(h)

def iir_filter(new_freqs_array, order):
  iir_freq = [new_freqs_array[2], new_freqs_array[4]]
  b, a = sgn.iirfilter(order, iir_freq, btype='bandpass')
  w, h = sgn.freqz(b, a)
  return b, a, w/np.pi, abs(h)


def main():
  freqs_array = [15, 30, 50, 75, 90]
  fs = 200
  time = 2
  t = np.linspace(0, time, time*fs)

  s = np.zeros(time*fs)
  for fr in freqs_array:
    s += np.sin(2*np.pi*fr*t)
  order = 7

  new_freqs_array = []
  new_freqs_array+=([i/200 for i in freqs_array])

  b_fir_fil, w_fir_fil, h_fir_fil = fir_filter(order, new_freqs_array)
  b_iir_fil, a_iir_fil, w_iir_fil, h_iir_fil = iir_filter(new_freqs_array, order)

  #ПОСЛЕДОВАТЕЛЬНОЕ СОЕДИНЕНИЕ
  consistent_filter = sgn.filtfilt(b_fir_fil, 1, sgn.filtfilt(b_iir_fil, a_iir_fil, s))

  plt.figure()
  plt.plot(s)
  plt.plot(consistent_filter)
  plt.show()

  #ГРАФИКИ ПЕРЕДАТОЧНОЙ ФУНКЦИИ FIR ФИЛЬТРА
  plt.figure()
  plt.plot(w_fir_fil, h_fir_fil)
  plt.show()
  #ГРАФИК ПЕРЕДАТОЧНОЙ ФУНКЦИИ IIR ФИЛЬТРА
  plt.figure()
  plt.plot(w_iir_fil, h_iir_fil)
  plt.show()
  #ГРАФИК ПЕРЕДАТОЧНОЙ ФУНКЦИИ ПОСЛЕДОВАТЕЛЬНОГО ФИЛЬТРА
  plt.figure()
  plt.plot(w_iir_fil, h_iir_fil*h_fir_fil)
  plt.show()
  #СПЕКТРОГРАММА
  consistent_filter = [int(consistent_filter[i]*200) for i in range(0, time*fs)]
  spectr_sin = rfft(consistent_filter)  # вычисляем дискретное действительное rfft  преобразование Фурье
  plt.figure()
  plt.plot(rfftfreq(time*fs, 1. / fs), abs(spectr_sin) / time*fs)  # график спектра
  plt.show()

if __name__ == '__main__':
    main()





