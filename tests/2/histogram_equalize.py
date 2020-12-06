import numpy as np


def equalize_histogram(histogram, max_shade=25):
    probabilities = histogram / sum(histogram)
    cummulative_probabilities = probabilities.cumsum()
    new_shades = [
        # It could be math.floor instead of round depending on the formula
        round(cummulative_probability * max_shade)
        for cummulative_probability in cummulative_probabilities
    ]

    return new_shades


X = -100
image = np.array([
    [3, 5, 6, 5, 3],
    [3, 6, 0, 5, 2],
    [5, 7, X, 2, 1],
    [6, 3, 4, 4, 0],
    [2, 7, 5, 2, 4],
])
# Discovers which value of X makes this image to be equalized to the matrix below

Y = -100
equalized_image = np.array([
    [11, 20, 23, 20, 11],
    [11, 23, 2, 20, 7],
    [20, 25, Y, 7, 3],
    [23, 11, 15, 15, 2],
    [7, 25, 20, 7, 15],
])

flatten_image = image.reshape(-1)  # Convert matrix to single array
x_index = np.argwhere(flatten_image == X)
flatten_image = np.delete(flatten_image, x_index)
shades, shades_count_arr = np.unique(flatten_image, return_counts=True)

flatten_equalized_image = equalized_image.reshape(-1)  # Convert matrix to single array
y_index = np.argwhere(flatten_equalized_image == Y)
flatten_equalized_image = np.delete(flatten_equalized_image, y_index)
equalized_shades = np.unique(flatten_equalized_image)

for x in range(len(shades_count_arr)):
    histogram_copy = shades_count_arr.copy()
    histogram_copy[x] += 1
    equalized_histogram = equalize_histogram(histogram_copy)
    print('x', x, 'equalized shades', equalized_histogram, end='')
    print(' found' if np.array_equal(equalized_histogram, equalized_shades) else '')
