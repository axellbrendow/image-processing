import sys
import numpy as np

if len(sys.argv) > 1:
    dft_inverse = [float(value) for value in sys.argv[1:]]
else:
    dft_inverse = [2+0j, 3+0j, 4+0j, 4+0j]

arr = np.array(dft_inverse)
dft_inverse = [value * len(arr) for value in np.fft.ifft(arr)]

print('discrete fourier transform', arr)
print('inverse discrete fourier transform multiplied by n:', dft_inverse)
