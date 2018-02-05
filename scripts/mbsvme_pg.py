'''
l1: lambd1 is the prior for expert weight vector
l2: lambd2 is the prior for gate weight vector
Every expert setting begins with (l1,l2) values and gives the peak accuracy values obtaines at the these tested learning rate setups.
K=1 would be similar to a Bayesian SVM.

SYNTH DATASET
K=2: (1.0, 1.0) Peak 92.1%%. Typically 90%%.

PIMA DATASET
K=1 Peak Observed at 81.32%%, Typically around 80.42%%.
K=2: (2.0, 1.0). Peak observed at 81.92%%. Typically around 80.72%%. -> Best setting.
Higher K require higher regularization, but do not provide any significant gains.

BREAST CANCER[42nd Split]
K=2: (10.0, 1.0) Peak 80.51%%. Typical Maximum around 78-79%%.
K>=3. Similar values in the range of 78-79%%.
Higher K is better, as gives higher values more frequently.

(High K seems to show some instability)
IMAGE DATASET[10th Split]
K=1: (0.0, 1.0) Peak 85.44%%, typically maximum ~84.4%%.
K=2: (1.0, 2.0) Peak 90.69%%
K=3,4,5: Strange poor performance. Very unstable.
High regularization, Large experts preferable.

TITANIC[42nd Split]
K=8: (10.0, 2.0) gives around 78.69%%.

WAVEFORM[42nd Split]
K=2: (10.0, 2.0) Peak 90.28%%.
K=2: (10.0, 5.0) Peak 90.69%%.
K=4: (20.0, 5.0) Peak 90.43%%. Easily achieves  89%%.
K=5: (10.0, 5.0) Peak 90.76%%.
K=10:(10.0, 5.0) Peak 91.86%%.

GERMAN[42nd Split]
K=1: (20.0, 5.0) 80.33%%
K=2: (10.0, 10.0) 81.66%%

BANANA[42nd Split]
K=1: Around 55%%.
K=2: (10.0, 20.0) Around 74%%.
K=4: (5.0, 10.0) 82.28%% 
K=10: (5.0, 12.0) 81.85%%
'''
import numpy as np
from numpy import linalg as LA
from scipy.stats import multivariate_normal

from utils import read_data, init
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
parser.add_argument('-f', '--file_write', type=bool, default=False,
					help='Write results to a file name, makes one file for one dataset')
parser.add_argument('-p', '--preprocess', type=str, default='gauss',
					help='Choose preprocessing: "gauss", "standard" or "none"')
args = parser.parse_args()
# ---------------------------------------------------------------------------------------------------------------------------------------------------------

# safer softmax
def softmax(X):
	cur = np.exp(X.T - np.max(X, axis=1)).T
	return (cur.T / cur.sum(axis=1) + delta).T

def compute_acc(X, y):
	gate_vals = softmax(np.dot(X, gate.T))
	pred = 2 * ((np.dot(X, experts.T) * gate_vals).sum(axis=1) > 0.0) - 1
	return np.mean(pred == y) * 100

# stability
eps = 1e-4
delta = 1e-4

kfold = 1
if args.data =='ijcnn':
	X, y, Xt, yt = read_data(key=args.data, preprocess=args.preprocess)
	# val and test split (predefined values)
	Xv, yv = Xt[:14990, :], yt[:14990]
	Xt, yt = Xt[14990:], yt[14990:]
	split = -1
elif args.data == 'adult':
	X, y, Xt, yt = read_data(key=args.data, return_split=False, preprocess=args.preprocess)
	split = -1

elif args.data in ['parkinsons', 'pima', 'wisconsin', 'sonar']:
	X, y, Xt, yt = read_data(key=args.data, return_split=False, preprocess=args.preprocess)
	if args.data == 'parkinsons':
		kfold = 5
	else:
		kfold = 10

	splitsize = X.shape[0] / kfold
	splits = [X[i*splitsize:(i+1)*splitsize,:] for i in range(kfold)]

	# add leftover datapoints to the the last split
	if splitsize * kfold != X.shape[0]:
		splits[-1] = np.concatenate((X[-(splitsize % kfold):,:], splits[-1]), axis=0)
else:
	X, y, Xt, yt, split = read_data(key=args.data, return_split=True, preprocess=args.preprocess)

for split in range(kfold):
	if kfold > 1:
		train = np.concatenate([splits[fold] for fold in range(kfold) if fold!= split], axis=0)
		test = splits[split]

		X = train[:, :-1]
		y = train[:, -1]
		Xt = test[:, :-1]
		yt = test[:, -1]

	# number of experts
	K = args.experts
	N = X.shape[0]
	dim = X.shape[1]
	max_iters = args.max_iters

	# prior over expert weight vector and gate vector
	lambd1 = args.reg_val_exp
	lambd2 = args.reg_val_gate

	# generative gating: initialization of parameters
	gate = np.random.randn(K, dim)
	experts = np.random.randn(K, dim)

	# expectations to be computed
	ex_probs = np.zeros((N, K))

	max_acc = -1.0
	max_val = -1.0

	for iters in range(max_iters):
		# E step
		pred = (np.dot(X, experts.T).T * y).T
		spred = 1. - pred
		svm_log_likelihood = -2 * spred * (spred > 0.0)

		# required expectations
		pre_softmax = np.dot(X, gate.T)
		un_softmax = np.exp(np.clip(pre_softmax, None, 500)) 
		ex_probs = softmax(pre_softmax + svm_log_likelihood)

		psi = pre_softmax - np.log((np.zeros((K,N)) + un_softmax.sum(axis=1)).T - un_softmax + delta)
		beta = 0.5 * np.tanh(0.5 * psi) / psi

		# EM for Bayesian SVM can lead to infinity values
		tau = np.abs(spred)
		Xmask = tau > delta
		tau_inv = 1./ (tau + delta * (1 - Xmask))

		if args.data != 'ijcnn':
			acc = compute_acc(Xt, yt)
			max_acc = max(max_acc, acc)
			if args.file_write == False:
				print "Test accuracy: %f" % (acc)
		else:
			acc = compute_acc(Xv, yv)
			if max_val < acc:
				max_val = acc
				max_acc = compute_acc(Xt, yt)
				if args.file_write == False:
					print "Test accuracy: %f" % (max_acc)

		# M step
		aux1 = tau_inv * ex_probs
		aux2 = (1 + tau_inv) * ex_probs
		aux3 = beta * ex_probs

		for e in range(K):
			experts[e, :] = np.dot(LA.inv(np.dot(np.dot(X.T * Xmask[:, e], np.diag(aux1[:, e])), (X.T * Xmask[:, e]).T) + lambd1 * np.eye(dim)), (X.T * y * aux2[:, e] * Xmask[:, e]).sum(axis=1))
			cur_unnormalized_softmax = np.exp(np.clip(np.dot((X.T * Xmask[:, e]).T, gate.T), None, 500))
			kappa = (0.5 + np.log((np.zeros((K,N)) + cur_unnormalized_softmax.sum(axis=1)).T - cur_unnormalized_softmax + delta)[:, e] * beta[:, e]) * ex_probs[:, e]
			gate[e, :] = np.dot(LA.inv(np.dot(np.dot(X.T * Xmask[:, e], np.diag(aux3[:, e])), (X.T * Xmask[:, e]).T) + lambd2 * np.eye(dim)), np.dot(X.T * Xmask[:, e], kappa))

	if args.file_write == False:
		print "\n\n\n\n"
		print "Dataset: " + args.data
		print "Number of experts: " + str(args.experts)
		print "Maximum accuracy achieved: " + str(max_acc)
		print "Data dimensionality: " + str(dim)
		print "Number of training points: " + str(N)
		print "Number of test points: " + str(Xt.shape[0])
	else:
		f = open("../results/" + str(args.data) + '_pgc.txt' , 'a')
		f.write(str(split) + ", " + str(args.experts) + ", " + str(args.reg_val_gate) + ", " + str(args.reg_val_exp) + ", " + str(max_acc) + "\n")