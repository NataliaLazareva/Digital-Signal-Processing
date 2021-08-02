#перевести строку с текстом в бинарное представление
#вычислить значения сплайна
#внедрить водяной знак в аудиофайл
#извлечь водяной знак из аудиофайла
import numpy as np
import scipy.io.wavfile as sw

msg = 'some string for watermark'
input = 'voice.wav'
fs, x  = sw.read(input)
#print(x.flags)
#x.flags.writeable = True

step = int(fs/100) #шаг внедрения = 160

def ConvertToBits():
 num = []
 for c in msg:
        bit = bin(ord(c))[2:]
        bit = '00000000'[len(bit):] + bit
        num.extend([int(b) for b in bit])
 print(len(num))
 return num

def BuildSpl(num):
 A = 0.5
 mas = np.linspace(0, 1, step)
 mas1 = mas[np.where(mas <= 0.33)]
 mas2 = mas[np.where((mas > 0.33) & (mas <= 0.66))]
 mas3 = mas[np.where(mas > 0.66)]

 spl = []
 spl.extend((9*mas1**2)/2)
 spl.extend(-9*mas2**2 + 9*mas2 - 3/2)
 spl.extend((9*(1 - mas3)**2)/2)

 u_plus = [(1 + i*A) for i in spl]
 u_minus = [(1 - i*A) for i in spl]

 return u_plus, u_minus

def EmbeddingWatermark(bits, u_plus, u_minus):
  p = position = 135*step
  x_WaterMark = x[::]
  #x_WaterMark.setflags(write=1, align=1)

  for bit in bits:
      if (bit == 0):
          for i in range(step): x_WaterMark[position+i] *= u_minus[i]
          
      else:
          for i in range(step): x_WaterMark[position + i] *= u_plus[i]

      position += step

  return np.int16(x_WaterMark), p

def ExtractWatermark(x_WaterMark, p):
        FindMarkBits = []
        for i in (bits):
            power_x = np.dot(x[p:(p+step)], x[p:p+step])
            power_x_WaterMark = np.dot(x_WaterMark[p:p+step], x_WaterMark[p:p+step])
            FindMarkBits.append(1 if power_x_WaterMark > power_x else 0)
            p+=step

        String = []
        for b in range(int(len(FindMarkBits) / 8)):
            byte = FindMarkBits[b * 8:(b + 1) * 8]
            String.append(chr(int(''.join([str(bit) for bit in byte]), 2)))
        return ''.join(String)

#шаг внедрения - длина фрагмента куда внедряется 1 бит, кол-во битов совп с количеством фрагментов

output = 'output.wav'

bits = ConvertToBits()
u_plus, u_minus = BuildSpl(bits)
x_WaterMark, p = EmbeddingWatermark(bits, u_plus, u_minus)

sw.write(output, fs, np.int16(x_WaterMark))

FindMark = ExtractWatermark(x_WaterMark, p)
print(msg, FindMark)