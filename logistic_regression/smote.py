import numpy as np
import matplotlib.pyplot as plt

x = np.array([[5, 3, 1], [6, 4, 3], [52, 21, 22], [502, 211, 524]])

plt.scatter(x[:, 0], x[:, 1])
plt.show()


class MinMaxNormalization:

    """
    axis:  0 is by columns, 1 is by rows
    """

    def __init__(self, data, axis=0):
        self.row_min = np.min(data, axis=axis)
        self.row_max = np.max(data, axis=axis)
        self.denominator = abs(self.row_max - self.row_min)

        # Fix divide by zero, replace value with 1 because these usually happen for boolean columns
        for index, value in enumerate(self.denominator):
            if value == 0:
                self.denominator[index] = 1

    def __call__(self, data):
        return np.divide((data - self.row_min), self.denominator)


normalizer = MinMaxNormalization(x)
x = normalizer(x)


def knn(data, distance):
    """
    Finds the k-nearest neighbours of a data set within a certain radius.
    Data fed in shape = [m, Nx]
    """

    d_matrix = np.zeros((data.shape[0], data.shape[0]))

    for i in range(data.shape[0]):
        for j in range(i):
            d = np.subtract(data[i, :], data[j, :])
            d = np.sum(np.power(d, 2))
            d_matrix[i, j] = np.sqrt(d)

    nearest_neighbours = []
    for i in range(d_matrix.shape[1]):
        for j in range(d_matrix.shape[0]):
            if d_matrix[j, i] == 0:
                pass
            elif d_matrix[j, i] < distance:
                nearest_neighbours.append([i, j])

    return nearest_neighbours, d_matrix

def smote(data, nearest_neighbours):
    pass

def adasyn(data, nearest_neighbours):
    pass
