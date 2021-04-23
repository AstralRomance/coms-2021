import math
import numpy as np

def print_matrix(filename: str, method_outp: tuple):
    with open(f'{filename}.txt', 'w') as outp:
        for matrix in method_outp:
            outp.write(f'{str(matrix)}\n')

def adamar_transform(array):
    

def reverse_adamar_transform(array: list):
    pass

def adamar_transform_sec(matrix: np.array):
    interim_matrix = 


signal = np.array([
    [0, 2, 3, 3, 3, 3, 2, 0],
    [0, 2, 1, 2, 1, -1, 0, 0],
    [0, 4, 3, 4, 3, 4, -1, 0],
    [0, 5, 5, 0, 4, 2, 1, 0],
    [6, 5, 3, 4, 6, 6, 0, 0],
    [0, -1, -2, -1, 2, 3, 5, 0],
    [0, 0, -1, 1, 2, 3, 0, 0],
    [0, 0, 2, 0, 3, 0, 3, 4]
])

noise_1 = np.array([
    [0, 2, 3, 3, 3, 3, 2, 0],
    [0, 2, 1, 2, 1, -1, 0, 0],
    [0, 4, 3, 4, 3, 4, -1, 0],
    [0, 5, 5, 9, 4, 2, 1, 0],
    [6, 5, 3, 4, 6, 6, 0, 0],
    [0, -1, -2, -1, 2, 3, 5, 0],
    [0, 0, -1, 1, 2, 1, 0, 0],
    [0, 0, 2, 0, 3, 0, 3, 4]
])

noise_2 = np.array([
    [0, 2, 3, 3, 3, 3, 2, 0],
    [0, 2, 1, 2, 1, -1, 0, 0],
    [0, 4, 3, 4, 3, 4, -1, 0],
    [0, 5, 5, 0, 4, 2, 1, 0],
    [6, 5, 3, 4, 6, 6, 0, 0],
    [0, -1, -2, -1, 2, 3, 5, 0],
    [0, 0, -1, 1, 2, 1, 9, 0],
    [0, 0, 2, 0, 3, 0, 3, 4]
])