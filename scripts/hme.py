import numpy as np
from numpy import linalg as LA

from utils import read_data

import argparse

# ---------------------------------------------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=str, default='banana',
					help='Name of the dataset')
parser.add_argument('-l','--levels', type=int, default=3,
					help='Number of levels in the tree, starting from 0')
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
args = parser.parse_args()
# ---------------------------------------------------------------------------------------------------------------------------------------------------------

# stability
eps = 0.

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
		splits[-1] = np.concatenate((X[-(splitsize % kfold):,:], splits[-1]), axis=0)

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

def compute_acc(X, y):
	ip = np.dot(X, W.T)
	
	gl = 1.0 + ip[:, :]
	gl = -2.0 * gl * (gl > 0.0)
	gr = 1.0 - ip[:, :]
	gr = -2.0 * gr * (gr > 0.0)

	for level in range(1, args.levels):
		for idx in range(2**level - 1, 2**(level + 1) - 1):
			if (idx - 1) % 2 == 1:
				gl[:, idx] += gr[:, (idx-1)/2]
				gr[:, idx] += gr[:, (idx-1)/2]
			else:
				gl[:, idx] += gl[:, (idx-1)/2]
				gr[:, idx] += gl[:, (idx-1)/2]
	
	# alternate prediction: normalization of gating values is optional, it does not change the prediction
	# if W.shape[0] > 1:
	# 	for idx in range(2**(args.levels - 1) - 1, 2**args.levels - 1):
	# 		print idx
	# 		if (idx - 1) % 2 == 1:
	# 			ip[:, idx] *= np.exp(gr[:, (idx-1)/2])
	# 		else:
	# 			ip[:, idx] *= np.exp(gl[:, (idx-1)/2])
	# pred = 2 * (ip[:, -K:].sum(axis=1) > 0.0) - 1
	
	# more mathematically justified
	pred = 2 * (np.exp(gl[:, -K:]).sum(axis=1) < np.exp(gr[:, -K:]).sum(axis=1)) - 1
	return np.mean(pred == y) * 100

for cur_fold in range(kfold):
	if kfold > 1:
		train = np.concatenate([splits[fold] for fold in range(kfold) if fold != cur_fold], axis=0)
		test = splits[cur_fold]

		split = cur_fold
		X = train[:, :-1]
		y = train[:, -1]
		Xt = test[:, :-1]
		yt = test[:, -1]

	# expert count
	K = 2 ** (args.levels-1)
	N = X.shape[0]
	dim = X.shape[1]
	max_iters = args.max_iters

	# prior over expert weight vector
	lambd = args.reg_value

	# weight matrix to be learnt
	W = np.random.randn(2*K-1, dim)
	
	max_acc = -1.0
	max_val = -1.0
	
	# expectations for E step
	ex_probs = np.zeros((N, 2*K - 1))
	
	for iters in range(max_iters):
		# precomputations
		ip = np.dot(X, W.T)
		gl_margin = 1.0 + ip[:, :-K]
		gl = -2.0 * gl_margin * (gl_margin > 0.0)
		gr_margin = 1.0 - ip[:, :-K]
		gr = -2.0 * gr_margin * (gr_margin > 0.0)

		for idx in range(K-1):
			ex_probs[:, idx*2 + 1] += ex_probs[:, idx] + gl[:, idx]
			ex_probs[:, idx*2 + 2] += ex_probs[:, idx] + gr[:, idx]

		# E step for leaf nodes/experts
		margin = 1. - (ip[:, -K:].T * y).T
		likelihood = -2.0 * margin * (margin > 0.0)
		ex_probs[:, -K:] = np.exp(ex_probs[:, -K:] + likelihood)
		ex_probs[:, -K:] = (ex_probs[:, -K:].T / ex_probs[:, -K:].sum(axis=1)).T

		tau_inv = 1./ (np.abs(margin) + eps)
		invtau_l = 1./ (np.abs(gl_margin) + eps)
		invtau_r = 1./ (np.abs(gr_margin) + eps)
		
		# predict accuracies
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

		# M step for leaf nodes/experts
		example_weights_mat = tau_inv * ex_probs[:, -K:]
		example_weights_vec = (1 + tau_inv) * ex_probs[:, -K:]
		for idx in range(K-1, 2*K-1):
			W[idx, :] = np.dot(LA.inv(np.dot(np.dot(X.T, np.diag(example_weights_mat[:, idx-K+1])), X) + lambd * np.eye(dim)), (X.T * y * example_weights_vec[:, idx-K+1]).sum(axis=1))
		
		# M step for non-leaf nodes
		for idx in reversed(range(K-1)):
			ex_probs[:, idx] = ex_probs[:, idx*2 + 1] + ex_probs[:, idx*2 + 2]
			example_weights_mat = ex_probs[:, idx*2 + 1] * invtau_l[:, idx] + ex_probs[:, idx*2 + 2] * invtau_r[:, idx]
			example_weights_vec = ex_probs[:, idx*2 + 2] * (1 + invtau_r[:, idx]) - ex_probs[:, idx*2 + 1] * (1 + invtau_l[:, idx])
			W[idx, :] = np.dot(LA.inv(np.dot(np.dot(X.T, np.diag(example_weights_mat)), X) + lambd * np.eye(dim)), (X.T * example_weights_vec).sum(axis=1))

	if args.file_write == False:
		print "\n\n\n\n"
		print "Dataset: " + args.data
		print "Number of levels: " + str(args.levels)
		print "Maximum accuracy achieved: " + str(max_acc)
		print "Data dimensionality: " + str(dim)
		print "Number of training points: " + str(N)
		print "Number of test points: " + str(Xt.shape[0])
	
	else:
		f = open("../results/" + str(args.data) + '_hmoe.txt' , 'a')
		f.write(str(split) + ", " + str(args.levels) + ", " + str(args.reg_value) + ", " + str(max_acc) + "\n")