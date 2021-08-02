#ЦВЗ на основе изменения магнитуды спектра.
#Формирование ЦВЗ, внедрение, обнаружение, атака на устойчивость в виде добавления шума.
import numpy as np
import scipy.io.wavfile as sw
import scipy.fftpack as sf
from random import gauss

def ConvertToBits(msg):
 num = []
 for c in msg:
        bit = bin(ord(c))[2:]
        bit = '00000000'[len(bit):] + bit
        num.extend([int(b) for b in bit])
 return np.array(num)

def EmbeddingWatermark(delta, start,  bin_watermark, sqn, x):
    eps = 0.05
    ep_0 = 1.0/(1.0+eps)
    ep_1 = 1.0/(1.0-eps)

    for bw in bin_watermark:
        fft_sqn = sf.fft(x[start:start+delta])

        #мощность исходной последовательности
        P = sum(pow(abs(fft_sqn), 2))
        #мощность N - k коэффициентов
        _P = P - fft_sqn[1] - fft_sqn[2] - fft_sqn[159] - fft_sqn[158]

        if bw == 0:
            fft_sqn[1] = abs(fft_sqn[1]) * ep_0
            fft_sqn[2] = abs(fft_sqn[2]) * ep_0
            fft_sqn[delta-1] = abs(fft_sqn[delta-1]) * ep_0
            fft_sqn[delta-2] = abs(fft_sqn[delta-2]) * ep_0
        else:
            fft_sqn[1] = abs(fft_sqn[1]) * ep_1
            fft_sqn[2] = abs(fft_sqn[2]) * ep_1
            fft_sqn[delta-1] = abs(fft_sqn[delta-1]) * ep_1
            fft_sqn[delta-2] = abs(fft_sqn[delta-2]) * ep_1

        # мощность полученной последовательности
        P2 = sum(pow(abs(fft_sqn), 2))
        delta_P = P - P2

        fft_sqn[3:delta-2] *= np.sqrt((_P + delta_P) / _P)

        fft_sqn = sf.ifft(fft_sqn)

        sqn[start:start+delta] = fft_sqn.real
        start += delta

    return sqn

def ExtractionWaterMark(sqn, WaterPlusSqn, start, delta):
    FindMarkBits = []

    while start + delta <= len(sqn):
        #вычисляем коэффициенты
        water_sqn_fft = sf.fft(WaterPlusSqn[start:start + delta])
        sqn_fft = sf.fft(sqn[start:start + delta])

        #Мощности первых двух коэффиуциентов
        P_wsf = sum(pow(abs(water_sqn_fft[1:3]), 2))
        P_sf = sum(pow(abs(sqn_fft[1:3]), 2))

        FindMarkBits.append(1 if P_wsf > P_sf else 0)

        start += delta
    print('извлеченные биты: ', FindMarkBits)


def main():
    input = 'voice.wav'
    fs, x = sw.read(input)
    watermark = 'some string for watermark'

    bin_watermark = ConvertToBits(watermark)

    sqn = np.zeros(len(x), dtype=int)
    delta = fs * 10 // 1000
    start = 555520
    sqn[0:start] = x[0:start]
    sqn[start + len(bin_watermark)*delta:] = x[start + len(bin_watermark)*delta:]

    WaterPlusSqn = EmbeddingWatermark(delta, start,  bin_watermark, sqn, x)

    ExtractionWaterMark(sqn, WaterPlusSqn, start, delta)
    print(watermark)

if __name__ == '__main__':
    main()