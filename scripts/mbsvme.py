'''
General softmax iterations: 10 steps along stepeest gradient with a learning rate of 0.01, unless specified otherwise
l1: lambd1 is the prior for expert weight vector
l2: lambd2 is the prior for gate weight vector
Every expert setting begins with (l1,l2) values and gives the peak accuracy values obtaines at the these tested learning rate setups.

Generally, with higher K, regularization becomes necessary for stable training.

IMAGE DATASET[10th Split]
K=1: (0.0, 2.0) Max: 85.14%%, typical maximum is 84.75%% Bayesian SVM Like
K=2: (0.0, 1.0) Typical ~91%%, maximum 92.17.
K=4: (0.3, 1.0) Typical ~90%%, maximum 92.57. 40 Softmax iterations at 0.1 learning rate.

SYNTH DATASET
K=1: Similar to Bayesian SVM.
K>=2: Maximum observed around 91.5%%.

BREAST CANCER[42nd Split]
'''
import argparse

# ---------------------------------------------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=str, default='banana',
					help='Name of the dataset')
parser.add_argument('-k','--experts', type=int, default=4,
					help='Number of Experts for the model')
parser.add_argument('-m', '--max_iters', type=int, default=50,
					help='Maximum number of iterations')
parser.add_argument('-re','--reg_val_exp', type=float, default=1.0,
					help='Regularization hyperparameter for prior on expert weight vectors')
parser.add_argument('-rg','--reg_val_gate', type=float, default=1.0,
					help='Regularization hyperparameter for prior on gating weight vectors')
parser.add_argument('-s', '--steps', type=int, default=10,
					help='Number of gradient descent steps for gating network')
parser.add_argument('-l','--lrate', type=float, default=0.05,
					help='Learning rate for gradient descent on gating network')
args = parser.parse_args()
# ---------------------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
from numpy import linalg as LA
from scipy.stats import multivariate_normal

from utils import read_data

# stability
eps = 1e-8
delta = 1e-12

X, y, Xt, yt = read_data(key=args.data)
# add preprocessing?

# number of experts 
K = args.experts
N = X.shape[0]
dim = X.shape[1]
max_iters = args.max_iters

# prior over expert weight vector and gate vector
lambd1 = args.reg_val_exp
lambd2 = args.reg_val_gate

# number of gradient descent steps per iteration
smax_iters_per_iter = args.steps
learning_rate = args.lrate

# generative gating: initialization of parameters
gate = np.random.randn(K, dim)
experts = np.random.randn(K, dim)

# expectations to be computed
ex_probs = np.zeros((N, K))

max_acc = -1.0

# safer softmax
def softmax(X):
	cur = np.exp(X.T - np.max(X, axis=1)).T
	return (cur.T / cur.sum(axis=1) + delta).T

def compute_acc():
	gate_vals = softmax(np.dot(Xt, gate.T))
	pred = 2 * ((np.dot(Xt, experts.T) * gate_vals).sum(axis=1) > 0.0) - 1

	return np.mean(pred == yt) * 100

for iters in range(max_iters):
	# E step
	pred = (np.dot(X, experts.T).T * y).T
	spred = 1. - pred
	svm_log_likelihood = -2 * spred * (spred > 0.0)

	# required expectations 
	ex_probs = softmax(np.dot(X, gate.T) + svm_log_likelihood)

	# EM for Bayesian SVM can lead to infinity values
	tau = np.abs(spred)
	Xmask = tau > delta
	tau_inv = 1./ (tau + delta * (1 - Xmask))

	acc = compute_acc()
	max_acc = max(max_acc, acc)
	print "Test accuracy: %f" % (acc)

	# M step
	aux1 = tau_inv * ex_probs
	aux2 = (1 + tau_inv) * ex_probs

	for e in range(K):
		experts[e, :] = np.dot(LA.inv(np.dot(np.dot(X.T * Xmask[:, e], np.diag(aux1[:, e])), (X.T * Xmask[:, e]).T) + lambd1 * np.eye(dim)), (X.T * y * aux2[:, e] * Xmask[:, e]).sum(axis=1))

	for _ in range(smax_iters_per_iter):
		cur_softmax = softmax(np.dot(X, gate.T))
		gradient = np.dot(X.T, ex_probs - cur_softmax).T - lambd2 * gate
		gate = gate - learning_rate * gradient

print "Maximum accuracy achieved: " + str(max_acc)