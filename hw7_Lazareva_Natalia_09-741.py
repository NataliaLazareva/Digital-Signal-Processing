#Внедрить ЦВЗ в сигнал, используя прямой и обратный IIR фильтры.
#Цифровой водяной знак определен методом главных компонент (PCA)
#Цифровой водяной знак добавляется методом ресширения спектра
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sgn
import scipy.io.wavfile as sw
import scipy.fftpack as sf
from random import gauss

def WatermarkPCA(x, m = 806, r = 403):
    i = 0
    Matr = []
    while i + m <= (len(x)):
             Matr.append(x[i:i+m])
             i+=r

    Matr = np.array(Matr)
    Matr = np.dot(Matr, Matr.T)
    w, v = np.linalg.eig(Matr)
    eigenvector = v[list(w).index(min(w))]
    return eigenvector

def iir_filter(x):
    freq = [4000/16000]
    order = 8
    #b, a = sgn.iirfilter(order, [4000], btype='high',  fs=16000)
    b, a = sgn.butter(order, freq, btype='high')
    return b, a

def filter(b, a, x):
    w, h = sgn.freqz(b, a)
    filter_sqn = sgn.filtfilt(b, a, x)
    return filter_sqn, w, h

def checking_b(b):
    coef = np.poly1d(b).roots
    return np.all(abs(coef) < 1)

def improving_b(b):
    #изменение коэффициентов b
    coef = list(np.poly1d(b).roots)
    for c in coef:
            if (abs(c) >= 1):
                coef[coef.index(c)] = complex(np.sqrt(c.real** 2 + 0.94 ** 2 - abs(c)** 2), c.imag)
                #coef[coef.index(c)] = complex(c.real / 100, c.imag)
    return np.poly1d(np.array(coef), True).coeffs

def EmbeddingWatermark(watermark, filter_sqn, p):
    coef = 0.3
    filter_sqn[p:p+len(watermark)] += coef*watermark
    return filter_sqn

def making_noise(x, watermark):
    noise = np.array([gauss(0.0, 1.0) for i in range(len(x))])
    x += noise
    corelation = np.correlate(x, watermark, 'valid')
    return corelation

def main():
    input = 'voice.wav'
    fs, x = sw.read(input)
    watermark = WatermarkPCA(x)
    b, a = iir_filter(x)
    filter_sqn, w, h = filter(b, a, x)

    # abs(coef) < 1?
    if (checking_b(b) == False):
       b = improving_b(b)
       filter_sqn, w, h = filter(b, a, x)

    p = 555594 # позиция внедрения
    watemark_sqn = EmbeddingWatermark(watermark, filter_sqn, p)
    len(watemark_sqn)
    reverse_sqn, w_reverse, h_reverse = filter(a, b, watemark_sqn)

    #ОБНАРУЖЕНИЕ ВОДЯНОГО ЗНАКА

    # КОРРЕЛЯЦИЯ С ФИЛЬТРАЦИЕЙ
    sqn, w, h = filter(b, a, watemark_sqn)
    cor_iir = np.correlate(sqn, watermark, 'valid')
    argmax = np.argmax(cor_iir)
    print('p_original_iir = ', p, 'p_cor_iir = ', argmax)

    # КОРРЕЛЯЦИЯ БЕЗ ПРЕДВ. ФИЛЬТРАЦИИ
    cor = np.correlate(watemark_sqn, watermark, 'valid')
    argmax = np.argmax(cor)
    print('p_original = ', p, 'p_cor = ', argmax)

    # ВЫВОД РЕЗУЛЬТАТОВ
    plt.figure()
    plt.plot(cor, 'purple')
    plt.title('График корреляции водяного знака без предварительной фильтрации')
    plt.show()

    plt.figure()
    plt.plot(cor_iir, 'yellow')
    plt.title('График корреляции водяного знака с предварительной фильтрацией')
    plt.show()

    plt.plot(x)
    plt.title('График исходного сигнала')
    plt.show()

    plt.plot(abs(sf.fft(x[:len(x)//2])))
    plt.title('Спектр исходного сигнала')
    plt.show()

    plt.plot(w / np.pi, abs(h))
    plt.title('Передаточная функция прямого фильтра')
    plt.show()

    plt.plot(w_reverse / np.pi, abs(h_reverse))
    plt.title('Передаточная функция обратного фильтра')
    plt.show()

    plt.plot(abs(sf.fft(filter_sqn[:len(filter_sqn)//2])))
    plt.title('Спектр сигнала после фильтрации')
    plt.show()

    # АТАКИ
    plt.figure()
    plt.plot(making_noise(watemark_sqn, watermark), 'green')
    plt.title('Корреляция после добавления шума')
    plt.show()

    # КОНВЕРТАЦИЯ СИГНАЛА
    output = 'convert.wav'
    sw.write(output, fs, np.int16(watemark_sqn))

if __name__ == '__main__':
    main()