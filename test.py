import numpy as np


# Min max normalization
def min_max_normalization(data):

    col_max = np.max(data, axis=1).reshape(data.shape[0], 1)
    col_min = np.min(data, axis=1).reshape(data.shape[0], 1)

    denominator = abs(col_max - col_min)

    # Fix divide by zero, replace value with 1 because these usually happen for boolean columns
    for index, value in enumerate(denominator):
        if value[0] == 0:
            print(index)
            denominator[index] = 1

    return np.divide((data - col_max), denominator), [col_max, col_min], denominator


# Define Normalization
def normalization(data):
    col_mean = np.mean(data, axis=1).reshape(data.shape[0], 1)
    col_std = np.std(data, axis=1).reshape(data.shape[0], 1)

    # Fix divide by 0 since sometimes, standard deviation can be zero
    for index, value in enumerate(col_std):
        if value[0] == 0:
            col_std[index] = 1

    return np.divide((data - col_mean), col_std)


# Modified Normalization robust to outliers
def mod_normalization(data):
    col_median = np.median(data, axis=1).reshape(data.shape[0], 1)
    mad = np.median(abs(data - col_median), axis=1).reshape(data.shape[0], 1)
    print(col_median, mad)

    # Fix divide by 0 for when MAD = 0
    for index, value in enumerate(mad):
        if value[0] == 0:
            mad[index] = 1

    return np.divide(0.6745 * (data - col_median), mad)


def important_features(weights, feature_list, threshold):
    """
    Returns all the important features above the threshold
    """

    index = np.linspace(0, weights.shape[0], weights.shape[0] + 1)
    index = [int(i) for (i, j) in zip(index, weights) if abs(j) > threshold]
    feature_list = [feature_list[i] for i in index]

    return feature_list


feature_list = ['chicken', 'beef', 'pork', 'lion', 'deer', 'dog']
weights = np.array([5, 1, 1.5, 2, 3, 5]).reshape(6, 1)

A = np.array([[1, 24, 421, 421, 0], [1, 42, 321, 422, 1], [1, 33, 355, 430, 0]]).T

