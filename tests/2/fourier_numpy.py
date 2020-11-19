import sys
import numpy as np

if len(sys.argv) > 1:
    imagem = [float(value) for value in sys.argv[1:]]
else:
    imagem = [2+0j, 3+0j, 4+0j, 4+0j]

arr = np.array(imagem)
dft_divided_by_n = [value / len(imagem) for value in np.fft.fft(imagem)]

print('imagem', imagem)
print('discrete fourier transform divided by n:', dft_divided_by_n)
