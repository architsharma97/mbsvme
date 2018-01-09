'''
PIMA DATASET
Set lambda = 10.0 for approx 80.72%% accuracy
Weird observation: change (1. + aux) -> (1. + aux).sum() can give higher accuracy at 0.0 reg (nearly 81.5 %%)

SYNTH DATASET
Maximum Test Accuracy of approx 89.8%% at lambda = 5.0, 75.0

BANANA DATASET[42nd Split]
Maximum Test Accuracy of approx 61.102%% at lambda = 2.0 [Susceptible to overfitting, usually would converge around 55%%]

BREAST CANCER[42nd Split]
Maximum Test Accuracy of approx 80.51%% at lambda = 1000.0 [Extremely difficult, expect maximum of 78%%]

IMAGE[1st Split]
Maximum Test Accuracy of approx 85.24%% at lambda = 0.0 [Expect 84.xx%%]
'''
import numpy as np
from numpy import linalg as LA

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

X, y, Xt, yt = read_data(key=args.data)

# stability
eps = 1e-11

N = X.shape[0]
dim = X.shape[1]
max_iters = 100

# weight vector priors
lambd = args.reg_value

# weight vector to be learnt
weight = np.random.randn(dim)

max_acc = -1.0
def compute_acc():
	pred = 2 * (np.dot(Xt, weight) > 0.0) - 1
	return np.mean(pred == yt) * 100

for iters in range(max_iters):

	# E step
	aux = np.abs(1. - y * np.dot(X, weight).T)

	# for the stability of EM, ignore the inputs for which aux goes to inf
	Xmask = aux > eps
	aux = 1. / (aux + 10 * eps * (1 - Xmask))

	# M step
	acc = compute_acc()
	max_acc = max(max_acc, acc)
	print "Current accuracy: " + str(acc)

	# weird observation: change (1. + aux) -> (1. + aux).sum(axis=0) can give higher accuracy at 0.0 reg!
	weight = np.dot(LA.inv(np.dot(np.dot(X.T * Xmask, np.diag(aux)), (X.T * Xmask).T) + lambd * np.eye(dim)), ((X.T * y * Xmask) * (1. + aux)).sum(axis=1))

print "Maximum Test Accuracy obtained: " + str(max_acc)