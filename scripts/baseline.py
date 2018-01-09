'''
PIMA DATASET
Without Kernel: Maximum test accuracy of 80.12%% at lambda = 5.0
With Kernel (2nd order polynomial): Maximum test accuracy of 80.42%% at lambda = 0.0
It is slightly unfair to use a kernel, because all other methods are not using kernels, and things usually become easier to resolve in higher dimensions.

SYNTH DATASET [No Kernels]
Max test accuracy of 89.7%% at lambda = 100.0

BANANA DATASET [42nd Split]
Max test accuracy of 57.551%% at lambda = 10.0

BREAST CANCER [42nd Split]
Max test accuracy of 76.62%% at lambda = 65.0

IMAGE [1st Split]
Max test accuracy of 82.67%% at lambda = 0.0
'''
import numpy as np
from utils import read_data
import argparse

# ---------------------------------------------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=str, default='banana',
					help='Name of the dataset')
parser.add_argument('-r', '--reg_value', type=float, default=1.0,
					help='Value of the regularization constant')
args = parser.parse_args()
# ---------------------------------------------------------------------------------------------------------------------------------------------------------

# arbitrary kernel : [1, x, y] -> [1, x, y, x^2, y^2, xy] 
def kernelize(X):
	return np.concatenate((X, X[:, 1:] ** 2, np.reshape(X[:, 1] * X[:, 2], (X.shape[0], 1))), axis=1)

X, y, Xt, yt = read_data(key=args.data)
lambd = args.reg_value

# X = kernelize(X)
# Xt = kernelize(Xt)

# linear classifier
w = np.dot(np.linalg.inv(np.dot(X.T, X) + lambd * np.eye(X.shape[1])), np.dot(X.T, y))

print 'Train: %f' %(np.mean(2 * (np.dot(X, w) > 0) - 1 == y) * 100)
print 'Test: %f' %(np.mean(2 * (np.dot(Xt, w) > 0) - 1 == yt) * 100)
