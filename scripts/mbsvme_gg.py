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

from utils import read_data, preprocessing

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
parser.add_argument('-f', '--file_write', type=bool, default=False,
					help='Write results to a file name, makes one file for one dataset')
parser.add_argument('-p', '--preprocess', type=str, default='gauss',
					help='Choose preprocessing: "gauss", "standard" or "none"')
parser.add_argument('-t', '--task', type=int, default=None,
					help='Choose the task id upon which this single task model is tested')
parser.add_argument('-i', '--init', type=str, default='random',
					help='Choose the type of initialization for Gaussian Gating')
args = parser.parse_args()
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
def compute_acc(X, y):
	gate_vals = np.zeros((X.shape[0], K))
	for e in range(K):
		gate_vals[:, e] = gate_prior[e] * multivariate_normal.pdf(X, mean=gate_mean[e, :], cov=gate_covariance[e, :, :])
	gate_vals = (gate_vals.T / (gate_vals.sum(axis=1) + eps)).T
	
	# pred = 2 * ((np.dot(X, experts.T) * gate_vals).sum(axis=1) > 0.0) - 1
	ln = 1. + np.dot(X, experts.T)
	ln = (np.exp(-2. * ln * (ln > 0.0)) * gate_vals).sum(axis=1)
	lp = 1. - np.dot(X, experts.T)
	lp = (np.exp(-2. * lp * (lp > 0.0)) * gate_vals).sum(axis=1)
	pred = 2 * (ln < lp) - 1

	return np.mean(pred == y) * 100

# stability
eps = 1e-4
delta = 1e-4

kfold = 1
if args.data == 'ijcnn':
	X, y, Xt, yt = read_data(key=args.data, preprocess=args.preprocess)
	# val and test split (predefined values)
	Xv, yv = Xt[:14990, :], yt[:14990]
	Xt, yt = Xt[14990:], yt[14990:]
	split = -1

elif args.data in ['adult', 'landmine_c', 'mnist_c', 'sentiment_c']:
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
		splits[-1] = np.concatenate((X[-(X.shape[0] % kfold):,:], splits[-1]), axis=0)

elif args.data in ['landmine', 'mnist', 'sentiment']:
	X, y, Xt, yt = read_data(key=args.data, return_split=False, preprocess=args.preprocess)
	
	task_num = {'landmine': 19, 'mnist': 10, 'sentiment': 4}

	# train on each as a single task model
	if args.task is None:
		split = taskid = np.random.randint(0, task_num[args.data])
	else:
		split = taskid = args.task

	X, y, Xt, yt = X[taskid], y[taskid], Xt[taskid], yt[taskid]

else:
	X, y, Xt, yt, split = read_data(key=args.data, return_split=True, preprocess=args.preprocess)

for cur_fold in range(kfold):
	if kfold > 1:
		train = np.concatenate([splits[fold] for fold in range(kfold) if fold != cur_fold], axis=0)
		test = splits[cur_fold]

		split = cur_fold
		X = train[:, :-1]
		y = train[:, -1]
		Xt = test[:, :-1]
		yt = test[:, -1]

		X, m, std = preprocessing(X, preprocess=args.preprocess, return_stats=True)
		X, y, Xt, yt = X, np.array(y), (np.array(Xt) - m) / std, np.array(yt)

	# number of experts 
	K = args.experts
	N = X.shape[0]
	dim = X.shape[1]
	max_iters = args.max_iters

	# prior over expert weight vector
	lambd = args.reg_value

	# generative gating: initialization of parameters
	if args.init == 'random':
		experts = np.random.randn(K, dim)
		gate_mean = np.random.randn(K, dim)
		gate_prior = np.random.dirichlet([1] * K)
		gate_covariance = np.array([np.diag(np.random.uniform(0.0, 1.0, dim))] * K)
	
	# expectations to be computed
	ex_probs = np.zeros((N, K))
	gate_probs = np.zeros((N, K))

	max_acc = -1.0
	max_val = -1.0
		
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
		tau_inv = 1./ (tau + delta)

		# expected complete log likelihood
		# expected_cll = 0.5 * lambd * LA.norm(experts)**2 + (ex_probs * (pred * (tau_inv + 2) - 0.5 * pred**2 + np.log(gate_probs))).sum()
		# print "E[CLL]: %f" %(expected_cll), 
		
		if args.data != 'ijcnn':
			acc = compute_acc(Xt, yt)
			max_acc = max(max_acc, acc)
			if args.file_write == False:
				print("Test accuracy: %f" % (acc))
		else:
			acc = compute_acc(Xv, yv)
			if max_val < acc:
				max_val = acc
				max_acc = compute_acc(Xt, yt)
				if args.file_write == False:
					print("Test accuracy: %f" % (max_acc))

		# M step
		aux1 = tau_inv * ex_probs
		aux2 = (1 + tau_inv) * ex_probs
		Nj = ex_probs.sum(axis=0)
		gate_prior = Nj / N

		for e in range(K):
			# experts[e, :] = np.dot(LA.inv(np.dot(np.dot(X.T * Xmask[:, e], np.diag(aux1[:, e])), (X.T * Xmask[:, e]).T) + lambd * np.eye(dim)), (X.T * y * aux2[:, e] * Xmask[:, e]).sum(axis=1))
			R = LA.cholesky(np.dot(X.T * aux1[:, e], X) + lambd * np.eye(dim))
			experts[e, :] = LA.solve(np.transpose(R), LA.solve(R, (X.T * y * aux2[:, e]).sum(axis=1)))
			
			gate_covariance[e, :, :] = np.dot((X - gate_mean[e, :]).T * ex_probs[:, e], (X - gate_mean[e, :])) / Nj[e] + eps * np.eye(dim)
			gate_mean[e, :] = (X.T * ex_probs[:, e]).sum(axis=1) / Nj[e]

	if args.file_write == False:
		print("\n\n\n\n")
		print("Dataset: " + args.data)
		print("Number of experts: " + str(args.experts))
		print("Maximum accuracy achieved: " + str(max_acc))
		print("Data dimensionality: " + str(dim))
		print("Number of training points: " + str(N))
		print("Number of test points: " + str(Xt.shape[0]))
	
	else:
		f = open("../results/" + str(args.data) + '_gg.txt' , 'a')
		f.write(str(split) + ", " + str(args.experts) + ", " + str(args.reg_value) + ", " + str(max_acc) + "\n")