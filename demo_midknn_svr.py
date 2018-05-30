# -*- coding: utf-8 -*- %reset -f
"""
@author: Hiromasa Kaneko
Demonstration of SVR hyperparameter optimization using the midpoints between
k-nearest-neighbor data points of a training dataset (midknn)
as a validation dataset in regression
"""

import time

import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split

import midknn

# Settings
k = 10  # k in k-nearest-neighbor algorithm
svr_cs = 2 ** np.arange(-5, 10, dtype=float)  # Candidates of C
svr_epsilons = 2 ** np.arange(-10, 0, dtype=float)  # Candidates of epsilon
svr_gammas = 2 ** np.arange(-20, 10, dtype=float)  # Candidates of gamma
number_of_training_samples = 300
number_of_test_samples = 100

# Generate samples for demonstration
X, y = datasets.make_regression(n_samples=number_of_training_samples + number_of_test_samples, n_features=10,
                                n_informative=10, noise=10, random_state=0)

# Divide samples into training samples and test samples
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=number_of_test_samples, random_state=0)

# Standarize X and y
autoscaled_Xtrain = (Xtrain - Xtrain.mean(axis=0)) / Xtrain.std(axis=0, ddof=1)
autoscaled_ytrain = (ytrain - ytrain.mean()) / ytrain.std(ddof=1)
autoscaled_Xtest = (Xtest - Xtrain.mean(axis=0)) / Xtrain.std(axis=0, ddof=1)

# Measure time in hyperparameter optimization
start_time = time.time()

# Optimize gamma by maximizing variance in Gram matrix
numpy_autoscaled_Xtrain = np.array(autoscaled_Xtrain)
variance_of_gram_matrix = list()
for svr_gamma in svr_gammas:
    gram_matrix = np.exp(
        -svr_gamma * ((numpy_autoscaled_Xtrain[:, np.newaxis] - numpy_autoscaled_Xtrain) ** 2).sum(axis=2))
    variance_of_gram_matrix.append(gram_matrix.var(ddof=1))
optimal_svr_gamma = svr_gammas[np.where(variance_of_gram_matrix == np.max(variance_of_gram_matrix))[0][0]]

# Optimize C and epsilon with midknn
midknn_index = midknn.midknn(autoscaled_Xtrain, k)  # generate indexes of midknn
X_midknn = (autoscaled_Xtrain[midknn_index[:, 0], :] + autoscaled_Xtrain[midknn_index[:, 1], :]) / 2
y_midknn = (ytrain[midknn_index[:, 0]] + ytrain[midknn_index[:, 1]]) / 2
r2_midknns = np.empty((len(svr_cs), len(svr_epsilons)))
rmse_midknns = np.empty((len(svr_cs), len(svr_epsilons)))
for svr_c_number, svr_c in enumerate(svr_cs):
    for svr_epsilon_number, svr_epsilon in enumerate(svr_epsilons):
        regression_model = svm.SVR(kernel='rbf', C=svr_c, epsilon=svr_epsilon, gamma=optimal_svr_gamma)
        regression_model.fit(autoscaled_Xtrain, autoscaled_ytrain)
        estimated_y_midknn = np.ndarray.flatten(regression_model.predict(X_midknn))
        estimated_y_midknn = estimated_y_midknn * ytrain.std(ddof=1) + ytrain.mean()
        r2_midknns[svr_c_number, svr_epsilon_number] = float(
            1 - sum((y_midknn - estimated_y_midknn) ** 2) / sum((y_midknn - y_midknn.mean()) ** 2))
        rmse_midknns[svr_c_number, svr_epsilon_number] = float(
            (2 * (len(ytrain) + 1) * sum((y_midknn - estimated_y_midknn) ** 2) / len(ytrain) / (
                        len(y_midknn) - 1)) ** 0.5)

optimal_svr_c_epsilon_index = np.where(r2_midknns == r2_midknns.max())
optimal_svr_c = svr_cs[optimal_svr_c_epsilon_index[0][0]]
optimal_svr_epsilon = svr_epsilons[optimal_svr_c_epsilon_index[1][0]]

# Check time in hyperparameter optimization
elapsed_time = time.time() - start_time

# Check optimized hyperparameters
print("C: {0}, Epsion: {1}, Gamma: {2}".format(optimal_svr_c, optimal_svr_epsilon, optimal_svr_gamma))

# Construct SVR model
regression_model = svm.SVR(kernel='rbf', C=optimal_svr_c, epsilon=optimal_svr_epsilon, gamma=optimal_svr_gamma)
regression_model.fit(autoscaled_Xtrain, autoscaled_ytrain)

# Calculate y of training dataset
calculated_ytrain = np.ndarray.flatten(regression_model.predict(autoscaled_Xtrain))
calculated_ytrain = calculated_ytrain * ytrain.std(ddof=1) + ytrain.mean()
# r2, RMSE, MAE
print("r2: {0}".format(float(1 - sum((ytrain - calculated_ytrain) ** 2) / sum((ytrain - ytrain.mean()) ** 2))))
print("RMSE: {0}".format(float((sum((ytrain - calculated_ytrain) ** 2) / len(ytrain)) ** 0.5)))
print("MAE: {0}".format(float(sum(abs(ytrain - calculated_ytrain)) / len(ytrain))))
# yy-plot
plt.figure(figsize=figure.figaspect(1))
plt.scatter(ytrain, calculated_ytrain)
YMax = np.max(np.array([np.array(ytrain), calculated_ytrain]))
YMin = np.min(np.array([np.array(ytrain), calculated_ytrain]))
plt.plot([YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin)],
         [YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin)], 'k-')
plt.ylim(YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin))
plt.xlim(YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin))
plt.xlabel("Actual Y")
plt.ylabel("Calculated Y")
plt.show()

# Estimate y of midknn
estimated_y_midknn = np.ndarray.flatten(regression_model.predict(X_midknn))
estimated_y_midknn = estimated_y_midknn * ytrain.std(ddof=1) + ytrain.mean()
# r2cv, RMSEcv, MAEcv
print("r2midknn: {0}".format(
    float(1 - sum((y_midknn - estimated_y_midknn) ** 2) / sum((y_midknn - y_midknn.mean()) ** 2))))
print("RMSEmidknn: {0}".format(
    float((2 * (len(ytrain) + 1) * sum((y_midknn - estimated_y_midknn) ** 2) / len(ytrain) / (
                len(y_midknn) - 1)) ** 0.5)))
print("MAEmidknn: {0}".format(float(sum(abs(y_midknn - estimated_y_midknn)) / len(y_midknn) * (
        2 * (len(ytrain) + 1) / len(ytrain) * len(y_midknn) / (len(y_midknn) - 1)) ** 0.5)))
# yy-plot
plt.figure(figsize=figure.figaspect(1))
plt.scatter(y_midknn, estimated_y_midknn)
YMax = np.max(np.array([np.array(y_midknn), estimated_y_midknn]))
YMin = np.min(np.array([np.array(y_midknn), estimated_y_midknn]))
plt.plot([YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin)],
         [YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin)], 'k-')
plt.ylim(YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin))
plt.xlim(YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin))
plt.xlabel("Actual Y")
plt.ylabel("Estimated Y of midknn")
plt.show()

# Estimate y of test dataset
predicted_ytest = np.ndarray.flatten(regression_model.predict(autoscaled_Xtest))
predicted_ytest = predicted_ytest * ytrain.std(ddof=1) + ytrain.mean()
# r2p, RMSEp, MAEp
print("r2p: {0}".format(float(1 - sum((ytest - predicted_ytest) ** 2) / sum((ytest - ytest.mean()) ** 2))))
print("RMSEp: {0}".format(float((sum((ytest - predicted_ytest) ** 2) / len(ytest)) ** 0.5)))
print("MAEp: {0}".format(float(sum(abs(ytest - predicted_ytest)) / len(ytest))))
# yy-plot
plt.figure(figsize=figure.figaspect(1))
plt.scatter(ytest, predicted_ytest)
YMax = np.max(np.array([np.array(ytest), predicted_ytest]))
YMin = np.min(np.array([np.array(ytest), predicted_ytest]))
plt.plot([YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin)],
         [YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin)], 'k-')
plt.ylim(YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin))
plt.xlim(YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin))
plt.xlabel("Actual Y")
plt.ylabel("Estimated Y")
plt.show()
