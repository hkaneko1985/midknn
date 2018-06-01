# indexes of the midpoints between k-nearest-neighbor data points
# of a training dataset (midknn) as a validation dataset in regression
import numpy as np


# calculate indexes of the midpoints between k-nearest-neighbor data points
# of a training dataset (midknn) as a validation dataset in regression

# X: array(numpy) of explanatory variables
# k : k in k-nearest-neighbor algorithm
def midknn(x_variables, k):
    sample_pair_numbers = np.empty((x_variables.shape[0], k))
    for sample_number in range(0, x_variables.shape[0]):
        distance = ((x_variables - x_variables[sample_number, :]) ** 2).sum(axis=1) ** 0.5
        distance_order = np.argsort(distance)
        sample_pair_numbers[sample_number, :] = distance_order[1:k + 1]

    midknn_index = np.empty((x_variables.shape[0] * k, 2), dtype='int64')
    for nearest_sample_number in range(0, k):
        midknn_index[nearest_sample_number * x_variables.shape[0]:(nearest_sample_number + 1) * x_variables.shape[0], 0] =\
            np.arange(0, x_variables.shape[0])
        midknn_index[nearest_sample_number * x_variables.shape[0]:(nearest_sample_number + 1) * x_variables.shape[0], 1] =\
            sample_pair_numbers[:, nearest_sample_number]

    return midknn_index
