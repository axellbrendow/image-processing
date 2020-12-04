import numpy as np
from scipy import signal

image = np.array([
    [4, 0, 2, 4, 4]],
    dtype='float'
)

kernel = np.array([
    [4, 4, 1, 4]],
    dtype='float'
)

result = signal.convolve2d(image, kernel, mode='full', boundary='fill')
print('all positions convolve2d:')
print(result)

result = signal.convolve2d(image, kernel, mode='valid', boundary='fill')
print('only full overlap convolve2d:')
print(result)
