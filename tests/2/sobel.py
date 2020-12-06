import numpy
import scipy.signal


def sobel(image):
    filter_x = scipy.signal.convolve2d(
        image,
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        boundary='wrap',
        mode='valid'
    )

    filter_y = scipy.signal.convolve2d(
        image,
        [[-1, -2, -1],
         [+0, +0, +0],
         [+1, +2, +1]],
        boundary='wrap',
        mode='valid'
    )

    return numpy.add(numpy.abs(filter_x), numpy.abs(filter_y))


def printnp2d(arr):
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            print(f'{arr[j][i]:.2f}', end='\t')
        print()


X = -100
image = [
    [3, 3, 0, 2, 2],
    [2, 1, 4, 0, 3],
    [1, 2, X, 2, 4],
    [3, 3, 3, 0, 1],
    [4, 3, 4, 1, 2]
]

final_sobel_filtered_image = [
    [4, 4, 6],
    [6, 4, 6],
    [8, 14, 4]
]

while(True):
    imagem_with_sobel_filter = sobel(image)
    if((imagem_with_sobel_filter == final_sobel_filtered_image).all()):
        X = image[2][2]
        print('X =', X)
        break
    image[2][2] += 1

image[2][2] = X
printnp2d(numpy.transpose(sobel(image)))
