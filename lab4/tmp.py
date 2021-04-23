
import csv
import math
import numpy as np
import matplotlib.pyplot as plt


M = 8
ZERO = 0.1


def read_mat(file, dim, delimiter):
    mat = np.zeros((dim, dim))  # Initial matrix with noise

    with open(file) as matfile:
        dialect = csv.Dialect
        dialect.delimiter = delimiter
        dialect.lineterminator = '\n'
        dialect.quoting = 0
        dialect.quotechar = '\\'
        matreader = csv.reader(matfile, dialect=dialect)
        line = 0
        for row in matreader:
            if len(row) != dim:
                print('File: ' + file + ': Sequence of ' + str(dim) + ' numbers expected, got ' + str(len(row)))
                exit(1)
            for i in range(len(row)):
                mat[line][i] = row[i]
            line += 1
    if len(mat) != dim:
        print('File: ' + file + str(dim) + ' rows in matrix expected, got ' + str(len(mat)))

    return mat


def make_plots(mat, title):
    dim_x, dim_y = mat.shape
    x = np.arange(0, dim_x)
    y = np.arange(0, dim_y)
    x, y = np.meshgrid(x, y)
    z_max = np.max(mat)
    z_min = np.min(mat)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d', title=title)
    ax.plot_surface(x, y, mat, cmap='inferno', zorder=2)
    ax.invert_yaxis()
    if z_max < 10 and z_min >= 0:
        ax.set_zlim(0, 10)

    x = np.arange(0, dim_x)
    y = np.arange(0, dim_y)
    x, y = np.meshgrid(x, y)

    fig = plt.figure()
    ax2 = fig.add_subplot(title=title + ' (projection)')
    ax2.pcolormesh(x, y, mat, shading='auto')
    ax2.invert_yaxis()


def adamar_step(array, is_reversed):
    if len(array) % 2 != 0:
        print('Input signal for Adamar transform dimension should be power of two, not multiple of ' + str(len(array)))
    delta = len(array) // 2

    result = np.zeros(delta * 2)
    k = 1
    if is_reversed:
        k = -1

    for i in range(delta):
        result[i] = array[i] + k * array[i + delta]
        result[i + delta] = array[i] + (-k) * array[i + delta]
    return result


def adamar_transform_1(array, is_reversed):
    if len(array) == 1:
        return array
    else:
        result = adamar_step(array, is_reversed)
        dim = len(array) // 2
        return np.concatenate((adamar_transform_1(result[0:dim], False),
                              adamar_transform_1(result[dim:], True)), axis=None)


def get_permutation(power):
    dim = int(pow(2, power))
    return np.array(list([int('{:0{width}b}'.format(i, width=power)[::-1], base=2) for i in range(dim)]))


def get_rev_permutation(power):
    if power == 3:
        return np.array([0, 7, 4, 3, 2, 5, 6, 1])
    elif power == 2:
        return np.array([0, 3, 2, 1])
    elif power == 4:
        return np.array([0, 15, 8, 7, 6, 9, 14, 1, 2, 13, 10, 5, 4, 11, 12, 3])


def get_power(dim):
    power = math.log(dim, 2)
    if int(power) != power:
        print('Input signal for Adamar transform dimension should be power of two, not ' + str(dim))
        exit(-1)
    return int(power)


def adamar_transform(array):
    permuted_array = array[get_permutation(get_power(len(array)))]
    return adamar_transform_1(permuted_array, False)


def rev_adamar_transform(array):
    get_power(len(array))
    array = array / len(array)
    result = adamar_transform_1(array, False)
    return result[get_rev_permutation(get_power(len(result)))]


def adamar_transform_2(matrix):
    interim_mat = np.array(list(adamar_transform(matrix[i]) for i in range(np.shape(matrix)[0])))
    return np.array(list(adamar_transform(interim_mat[:, i]) for i in range(np.shape(interim_mat)[1])))


def rev_adamar_transform_2(matrix):
    interim_mat = np.array(list(rev_adamar_transform(matrix[i]) for i in range(np.shape(matrix)[0])))
    return np.array(list(rev_adamar_transform(interim_mat[:, i]) for i in range(np.shape(interim_mat)[1])))


def generate_filter(signal_spectrum, noise_spectrum):
    filter_s = np.divide(signal_spectrum, noise_spectrum)
    filter_s[filter_s == math.inf] = 0
    filter_s[filter_s == -math.inf] = 0
    print(filter_s)
    return filter_s


def main():
    signal = read_mat('signal.csv', M, delimiter=';')    # Initial matrix without noise
    print('Initial signal without noise:\n' + str(signal))
    make_plots(signal, 'Initial signal without noise')

    noise1 = read_mat('noise1.csv', M, delimiter=';')    # Signal with noise 1
    print('Signal with noise 1:\n' + str(noise1))
    make_plots(noise1, 'Signal with noise 1')

    noise2 = read_mat('noise2.csv', M, delimiter=';')    # Signal with noise 2
    print('Signal with noise 2:\n' + str(noise2))
    make_plots(noise2, 'Signal with noise 2')

    adamar_s = adamar_transform_2(signal)
    print('Adamar transform of initial signal:\n' + str(adamar_s))
    make_plots(adamar_s, 'Initial signal Adamar transform')
    np.savetxt("adamar_s.csv", adamar_s, delimiter=' ', fmt='%10.1f')

    adamar_1 = adamar_transform_2(noise1)
    print('Adamar transform of noise 1:\n' + str(adamar_1))
    make_plots(adamar_1, 'Noise 1 Adamar transform')
    np.savetxt("adamar_1.csv", adamar_1, delimiter=' ', fmt='%10.1f')

    adamar_2 = adamar_transform_2(noise2)
    print('Adamar transform of noise 2:\n' + str(adamar_2))
    make_plots(adamar_2, 'Noise 2 Adamar transform')
    np.savetxt("adamar_2.csv", adamar_2, delimiter=' ', fmt='%10.1f')

    filter_s = generate_filter(adamar_s, adamar_1)
    print('Filter for noise 1:\n' + str(filter_s))
    make_plots(filter_s, 'Filter')
    np.savetxt("filter_s.csv", filter_s, delimiter=' ', fmt='%10.1f')

    adamar_f1 = filter_s * adamar_1
    print('Filtered Adamar spectrum of noise 1:\n' + str(adamar_f1))
    make_plots(adamar_f1, 'Noise 1 filtered Adamar spectrum')
    np.savetxt("adamar_f1.csv", adamar_f1, delimiter=' ', fmt='%10.1f')

    adamar_f2 = filter_s * adamar_2
    print('Filtered Adamar spectrum of noise 2:\n' + str(adamar_f2))
    make_plots(adamar_f2, 'Noise 2 filtered Adamar spectrum')
    np.savetxt("adamar_f2.csv", adamar_f2, delimiter=' ', fmt='%10.1f')

    signal1 = rev_adamar_transform_2(adamar_f1)
    print('Filtered noise 1:\n' + str(signal1))
    make_plots(signal1, 'Filtered noise 1')
    np.savetxt("signal1.csv", signal1, delimiter=' ', fmt='%10.1f')

    signal2 = rev_adamar_transform_2(adamar_f2)
    print('Filtered noise 2:\n' + str(signal2))
    make_plots(signal2, 'Filtered noise 2')
    np.savetxt("signal2.csv", signal2, delimiter=' ', fmt='%10.1f')

    plt.show()


if __name__ == '__main__':
    main()
