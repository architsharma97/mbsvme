import numpy as np
from numpy import linalg as LA
from scipy.stats import multivariate_normal

from utils import read_data

import argparse

# ---------------------------------------------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=str, default='landmine',
					help='Name of the dataset')
parser.add_argument('-k','--experts', type=int, default=4,
					help='Number of Experts for the model')
parser.add_argument('-m', '--max_iters', type=int, default=50,
					help='Maximum number of iterations')
parser.add_argument('-r', '--reg_value', type=float, default=1.0,
					help='Regularization hyperparameter for prior on expert weight vectors')
parser.add_argument('-f', '--file_write', type=bool, default=False,
					help='Write results to a file name, makes one file for one dataset')
parser.add_argument('-p', '--preprocess', type=str, default='gauss',
					help='Choose preprocessing: "gauss", "standard" or "none"')
args = parser.parse_args()
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
def compute_acc(X, y, taskid):
	gate_vals = np.zeros((X.shape[0], K))
	for e in range(K):
		gate_vals[:, e] = gate_prior[taskid][e] * multivariate_normal.pdf(X, mean=gate_mean[taskid][e, :], cov=gate_covariance[taskid][e, :, :])
	gate_vals = (gate_vals.T / (gate_vals.sum(axis=1) + eps)).T
	pred = 2 * ((np.dot(X, experts.T) * gate_vals).sum(axis=1) > 0.0) - 1
	
	return np.mean(pred == y) * 100

# stability
eps = 1e-3
delta = 1e-3

if args.data in ['landmine', 'mnist', 'sentiment']:
	X, y, Xt, yt = read_data(key=args.data, return_split=False, preprocess=args.preprocess)
	Xcomb = np.concatenate(X, axis=0)
	ycomb = np.concatenate(y, axis=0)

# number of experts
K = args.experts
N = [x.shape[0] for x in X]
dim = X[0].shape[1]
max_iters = args.max_iters
tasks = len(X)

# prior over expert weight vector
lambd = args.reg_value

# generative gating: initialization of parameters
gate_mean = [np.random.randn(K, dim) for _ in range(tasks)] 
gate_prior = [np.random.dirichlet([1] * K) for _ in range(tasks)]
gate_covariance = [np.array([np.diag(np.random.uniform(0.0, 1.0, dim))] * K) for _ in range(tasks)]
experts = np.random.randn(K, dim)

# expectations
ex_probs = [np.zeros((n, K)) for n in N]
gate_probs = [np.zeros((n, K)) for n in N]

# auxiliary variables
tau = [None for _ in range(tasks)]
Xmask = [None for _ in range(tasks)]
tau_inv = [None for _ in range(tasks)]
aux1 = [None for _ in range(tasks)]
aux2 = [None for _ in range(tasks)]
Nj = [None for _ in range(tasks)]

# tracking accuracies
acc = [None for _ in range(tasks)]
max_acc = [-1.0 for _ in range(tasks)]
max_av_acc = -1.0

for iters in range(max_iters):
	
	# E step
	pred = [(np.dot(X[i], experts.T).T * y[i]).T for i in range(tasks)]
	spred = [1. - pred[i] for i in range(tasks)]
	svm_likelihood = [np.exp(-2 * spred[i] * (spred[i] > 0.0)) for i in range(tasks)]

	for i in range(tasks):
		for e in range(K):
			gate_probs[i][:, e] = multivariate_normal.pdf(X[i], mean=gate_mean[i][e, :], cov=gate_covariance[i][e, : ,:]) * gate_prior[i][e] + delta
			ex_probs[i][:, e] = svm_likelihood[i][:, e] * gate_probs[i][:, e]
	
		# required expectations
		ex_probs[i] = (ex_probs[i].T / ex_probs[i].sum(axis=1)).T
	
		# EM for Bayesian SVM can lead to infinity values
		tau[i] = np.abs(spred[i])
		Xmask[i] = tau[i] > delta
		tau_inv[i] = 1./ (tau[i] + delta * (1 - Xmask[i]))
 	
		acc[i] = compute_acc(Xt[i], yt[i], i)
		max_acc[i] = max(max_acc[i], acc[i])
		if args.file_write == False:
			print "Test accuracy for task %d: %f" % (i + 1, acc[i])

		# M step
		aux1[i] = tau_inv[i] * ex_probs[i]
		aux2[i] = (1 + tau_inv[i]) * ex_probs[i]
		Nj[i] = ex_probs[i].sum(axis=0)
		gate_prior[i] = Nj[i] / N[i]

		for e in range(K):
			gate_covariance[i][e, :, :] = np.dot(np.dot(((X[i].T * Xmask[i][:,e]).T - gate_mean[i][e, :]).T, np.diag(ex_probs[i][:, e])), ((X[i].T * Xmask[i][:, e]).T - gate_mean[i][e, :])) / Nj[i][e] + eps * np.eye(dim)
			gate_mean[i][e, :] = (X[i].T * ex_probs[i][:, e] * Xmask[i][:, e]).sum(axis=1) / Nj[i][e]
	
	max_av_acc = max(max_av_acc, np.mean(acc))
	Xmask_comb = np.concatenate(Xmask, axis=0)
	aux1_comb = np.concatenate(aux1, axis=0)
	aux2_comb = np.concatenate(aux2, axis=0)
	for e in range(K):
		experts[e, :] = np.dot(LA.inv(np.dot(np.dot(Xcomb.T * Xmask_comb[:, e], np.diag(aux1_comb[:, e])), (Xcomb.T * Xmask_comb[:, e]).T) + lambd * np.eye(dim)), (Xcomb.T * ycomb * aux2_comb[:, e] * Xmask_comb[:, e]).sum(axis=1))

if args.file_write == False:
	print "\n\n\n\n"
	print "Dataset: " + args.data
	print "Number of experts: " + str(args.experts)
	print "Maximum accuracy achieved individually: ", max_acc, np.mean(max_acc)
	print "Maximum average test accuracy achieved: ", max_av_acc

else:
	f = open("../results/" + str(args.data) + '_multitask_gg.txt' , 'a')
	f.write(str(split) + ", " + str(args.experts) + ", " + str(args.reg_value) + ", " + str(max_acc) + "\n")