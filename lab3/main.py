import math
import numpy as np
from view import make_colormap

def print_matrix(filename: str, method_outp: tuple):
    with open(f'{filename}.txt', 'w') as outp:
        for matrix in method_outp:
            outp.write(f'{str(matrix)}\n')


def anisotropic_filtering(source_matrix: np.array, filter_matrix: np.array, N=3, filter_coefficient=17) -> tuple:
    signal = np.zeros_like(source_matrix)
    signal_rows, signal_columns = source_matrix.shape

    for i in range(signal_rows):
        for j in range(signal_columns):
            if (i == 0) or (i == signal_rows-1) or (j == 0) or (j == signal_columns-1):
                signal[i][j] = source_matrix[i][j]
            else:
                outp_s = 0
                for k1 in range(N):
                    for k2 in range(N):
                        outp_s += filter_matrix[k1][k2] * source_matrix[i-1+k1][j-1+k2]
                    signal[i][j] = outp_s / filter_coefficient

    delta = source_matrix - signal
    return (signal, delta)

def static_filtering(source_matrix: np.array, N=3, m=1.23) -> tuple:
    signal = np.zeros_like(source_matrix)
    signal_rows, signal_columns = source_matrix.shape

    for i in range(signal_rows):
        for j in range(signal_columns):
            if (i == 0) or (i == signal_rows-1) or (j == 0) or (j == signal_columns-1):
                signal[i][j] = source_matrix[i][j]
            else:
                outp_s1 = 0
                outp_s2 = 0
                for k1 in range(N):
                    for k2 in range(N):
                        outp_s1 += source_matrix[i-1+k1][j-1+k2]
                G= outp_s1 / N**2

                for k1 in range(N):
                    for k2 in range(N):
                        outp_s2 += (source_matrix[i-1+k1][j-1+k2] - G)**2
                nu = m* math.sqrt(outp_s2/(N**2 - 1))
                if source_matrix[i][j] - G < nu:
                    signal[i][j] = source_matrix[i][j]
                else:
                    signal[i][j] = G
    delta = source_matrix - signal
    return (signal, delta)


source_matrix = np.array([
    [2,  6, 9, 11, 16, 24, 25, 43, 37, 44, 24, 41, 15, 16, 12], 
    [2, 8, 13, 21, 30, 56, 67, 101, 80, 88, 55, 53, 28, 20, 2], 
    [9, 24, 39, 55, 85, 100, 150, 164, 141, 99, 83, 64, 26, 33, 0], 
    [6, 26, 53, 82, 111, 119, 161, 161, 123, 101, 53, 77, 35, 29, 21], 
    [14, 39, 75, 112, 130, 129, 153, 152, 134, 109, 75, 82, 60, 28, 4], 
    [7, 27, 63, 107, 140, 133, 161, 146, 166, 119, 71, 47, 24, 40, 0], 
    [9, 31, 68, 118, 150, 150, 171, 158, 151, 121, 74, 14, -19, -18, 0], 
    [10, 32, 75, 128, 148, 250, 194, 221, 167, 152, 116, 70, 19, 36, 44], 
    [5, 32, 65, 116, 129, 171, 175, 182, 142, 250, 90, 81, 52, 41, 0], 
    [12, 33, 78, 106, 130, 133, 141, 155, 152, 90, 71, 72, 53, 45, 0], 
    [0, 7, 40, 71, 88, 103, 75, 87, 56, 61, 44, 31, 40, 18, 28], 
    [5, 8, 31, 50, 75, 84, 80, 74, 60, 73, 45, 11, -4, -11, 7], 
    [3, 1, 17, 17, 45, 55, 62, 60, 42, 45, 25, 16, 6, -1, 0], 
    [4, 1, 11, 15, 30, 43, 49, 57, 39, 43, 27, 18, 22, -11, 0], 
    [4, 10, 14, 20, 14, 34, 31, 70, 50, 57, 32, 26, 37, 21, 28]
])

filter_matrix = np.array([
                [2, 2, 1], 
                [2, 3, 2], 
                [1, 2, 2]
])

print_matrix('ans', anisotropic_filtering(source_matrix, filter_matrix))
make_colormap(source_matrix, 'source matrix')
make_colormap(anisotropic_filtering(source_matrix, filter_matrix)[1], 'Anisotropic filtering')
print_matrix('static', static_filtering(source_matrix))
make_colormap(static_filtering(source_matrix)[1], 'Static filtering')
