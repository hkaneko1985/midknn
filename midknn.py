# indexes of the midpoints between k-nearest-neighbor data points
# of a training dataset (midknn) as a validation dataset in regression
import numpy as np

# calculate indexes of the midpoints between k-nearest-neighbor data points
# of a training dataset (midknn) as a validation dataset in regression

# X: ndarray(numpy) of explanatory variables
# k : k in k-neaest-neighbor algorithm
def midknn( X, k ):
    samplepairnumbers = np.empty( (X.shape[0], k) )
    for samplenumber in range(0,X.shape[0]):
        distance = ((X - X[samplenumber,:])**2).sum(axis=1)**0.5
        distanceorder = np.argsort(distance)
        samplepairnumbers[samplenumber,:] = distanceorder[1:k+1]
    
    midknn_index = np.empty( (X.shape[0]*k, 2), dtype = 'int64' )
    for nearestsamplenumber in range(0,k):
        midknn_index[nearestsamplenumber*X.shape[0]:(nearestsamplenumber+1)*X.shape[0], 0] = np.arange(0,X.shape[0])
        midknn_index[nearestsamplenumber*X.shape[0]:(nearestsamplenumber+1)*X.shape[0], 1] = samplepairnumbers[:,nearestsamplenumber]

    return midknn_index
