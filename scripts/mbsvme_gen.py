'''
PIMA DATASET
Set K=1 gives back performance of Bayesian SVM (even though the objectives are slightly different). Use lambda = 10.0 for 80.72 %% accuracy.
Set K > 1, training is highly initialization dependent. Technically, K=1 is subsumed as a case in all K > 1, so the accuracy = 80.72%% is possible (and it can be encountered as well).
However, the training is unstable and can range in any results from 70-81 %%. Typical non-convex optimization issues [Also, higher accuracies are usually achieved when most of the
examples are assigned (even in prior) to one of the experts, indicating multiple experts are not being useful for the problem].

SYNTH DATASET
K=1: Behaves very similar to Bayesian SVM. Maximum accuracy at lambda = 5.0 of 89.8%%.
K=2: Lot of variations, but the Maximum Accuracy of 92.3%% is observed at lambda = 0.5 [This value is rare, mostly 91.5%% with decent regularization like 50.0].
K=3: Again lot of variations in test accuracies. Max observed test accuracy of 92.6%% at lambda = 0.5 [This achieves higher values more regularly, good regularization is 1.0]
K>=5: Similar.

BANANA DATASET [42nd Split]
K = 10: lambda = 5.0 gives a maximum test accuracy 90.57%%

BREAST CANCER [42nd Split]
In general, high regularization is necessary + reasonable number of experts
K=2: lambda = 500 gives a maximum test accuracy of 81.81%% [Extremely difficult]
K=4: lambda = 1000 gives 80.51%% with reasonable frequency [K=4 is a nice setting]

IMAGE [1st Split]
K=1: lambda = 0.0 gives Bayesian SVM kind of performance [84.xx %%]
K=2: lambda = 1.0 gives 92.77%%, but needs regularization
K=4: lambda = 1.0 gives 92.67%%
K=8: lambda = 2.0 gives 95.44%% + eps=1e-5 [Weird results, very difficult to achieve], achieved with eps=1e-8 but gives covariance related error frequently [Expect 92.xx%%]
K=20:lambda = 1.0 gives 96.13%% at eps=1e-6 [Not going to occur again it seems]

TITANIC[42nd Split] 
K=10: lambda = 1.0 gives 79.27%%.

WAVEFORM[42nd Split]
K=1: lambda = 75.0 gives 87.08%%
K=2 lambda = 15.0 gives around 85%% maximum, usually 83-84%% (extra experts do not seem to help)

GERMAN
K=1: lambda = 100.0 gives 82%%
K=2: lambda = 50.0 gives 81%% [Typically 79%% is achieved though]
'''
import numpy as np
from numpy import linalg as LA
from scipy.stats import multivariate_normal

from utils import read_data

import argparse

# ---------------------------------------------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=str, default='banana',
					help='Name of the dataset')
parser.add_argument('-k','--experts', type=int, default=4,
					help='Number of Experts for the model')
parser.add_argument('-m', '--max_iters', type=int, default=50,
					help='Maximum number of iterations')
parser.add_argument('-r', '--reg_value', type=float, default=1.0,
					help='Regularization hyperparameter for prior on expert weight vectors')
args = parser.parse_args()
# ---------------------------------------------------------------------------------------------------------------------------------------------------------

# stability
eps = 1e-6
delta = 1e-12

X, y, Xt, yt = read_data(key=args.data)
# add preprocessing?

# number of experts 
K = args.experts
N = X.shape[0]
dim = X.shape[1]
max_iters = args.max_iters

# prior over expert weight vector
lambd = args.reg_value

# generative gating: initialization of parameters
gate_mean = np.random.randn(K, dim)
gate_prior = np.random.dirichlet([1] * K)
gate_covariance = np.array([np.diag(np.random.uniform(0.0, 1.0, dim))] * K)
experts = np.random.randn(K, dim)

# expectations to be computed
ex_probs = np.zeros((N, K))
gate_probs = np.zeros((N, K))

max_acc = -1.0

def compute_acc():
	gate_vals = np.zeros((Xt.shape[0], K))
	for e in range(K):
		gate_vals[:, e] = gate_prior[e] * multivariate_normal.pdf(Xt, mean=gate_mean[e, :], cov=gate_covariance[e, :, :])
	gate_vals = (gate_vals.T / (gate_vals.sum(axis=1) + eps)).T
	pred = 2 * ((np.dot(Xt, experts.T) * gate_vals).sum(axis=1) > 0.0) - 1
	
	return np.mean(pred == yt) * 100

for iters in range(max_iters):
	
	# E step
	pred = (np.dot(X, experts.T).T * y).T
	spred = 1. - pred
	svm_likelihood = np.exp(-2 * spred * (spred > 0.0))

	for e in range(K):
		gate_probs[:, e] = multivariate_normal.pdf(X, mean=gate_mean[e, :], cov=gate_covariance[e, : ,:]) * gate_prior[e] + delta
		ex_probs[:, e] = svm_likelihood[:, e] * gate_probs[:, e]
	
	# required expectations
	ex_probs = (ex_probs.T / ex_probs.sum(axis=1)).T
	
	# EM for Bayesian SVM can lead to infinity values
	tau = np.abs(spred)
	Xmask = tau > delta
	tau_inv = 1./ (tau + delta * (1 - Xmask))

	# expected complete log likelihood
	# expected_cll = 0.5 * lambd * LA.norm(experts)**2 + (ex_probs * (pred * (tau_inv + 2) - 0.5 * pred**2 + np.log(gate_probs))).sum()
	# print "E[CLL]: %f" %(expected_cll), 
	
	acc = compute_acc()
	max_acc = max(max_acc, acc)
	print "Test accuracy: %f" % (acc)

	# M step
	aux1 = tau_inv * ex_probs
	aux2 = (1 + tau_inv) * ex_probs
	Nj = ex_probs.sum(axis=0)
	gate_prior = Nj / N

	for e in range(K):
		experts[e, :] = np.dot(LA.inv(np.dot(np.dot(X.T * Xmask[:, e], np.diag(aux1[:, e])), (X.T * Xmask[:, e]).T) + lambd * np.eye(dim)), (X.T * y * aux2[:, e] * Xmask[:, e]).sum(axis=1))
		gate_covariance[e, :, :] = np.dot(np.dot(((X.T * Xmask[:,e]).T - gate_mean[e, :]).T, np.diag(ex_probs[:, e])), ((X.T * Xmask[:, e]).T - gate_mean[e, :])) / Nj[e] + eps * np.eye(dim)
		gate_mean[e, :] = (X.T * ex_probs[:, e] * Xmask[:, e]).sum(axis=1) / Nj[e]

print "\n\n\n\n"
print "Dataset: " + args.data
print "Number of experts: " + str(args.experts)
print "Maximum accuracy achieved: " + str(max_acc)
print "Data dimensionality: " + str(dim)
print "Number of training points: " + str(N)
print "Number of test points: " + str(Xt.shape[0])