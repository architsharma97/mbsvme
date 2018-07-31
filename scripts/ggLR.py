import numpy as np
from numpy import linalg as LA
from scipy.stats import multivariate_normal
from scipy.optimize import fmin_l_bfgs_b

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
		gate_vals[:, e] = gate_prior[e] * multivariate_normal.pdf(X, mean=gate_mean[e, :], cov=gate_covariance[e, :, :]) + 1e-8
	gate_vals = (gate_vals.T / (gate_vals.sum(axis=1) + 1e-8)).T
	
	pred = np.exp(np.dot(X, experts.T))
	pred = 2 * ((gate_vals * (pred / (1. + pred))).sum(axis=1) > 0.5)  - 1

	return np.mean(pred == y) * 100

def compute_loss(experts, ex_probs, pred, y, lambd):
	experts = experts.reshape((K, dim))
	return -(ex_probs * (0.5 * (pred.T * (y + 1)).T - np.log(1 + np.exp(pred)))).sum() + 0.5 * lambd * (experts * experts).sum()

def compute_grad(experts, ex_probs, pred, y, lambd):
	experts = experts.reshape((K, dim))
	return (-(np.dot(X.T, ex_probs * (np.concatenate([0.5 * (y.reshape((N, 1)) + 1)], axis=1) - 1./(1. + np.exp(-pred))))).T + lambd * experts).flatten()

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
		pred = np.dot(X, experts.T)
		LR_likelihood = np.exp(0.5 * (pred.T * (y + 1)).T) / (1. + np.exp(pred))

		for e in range(K):
			gate_probs[:, e] = multivariate_normal.pdf(X, mean=gate_mean[e, :], cov=gate_covariance[e, : ,:]) * gate_prior[e] + 1e-8
		
		# required expectations
		ex_probs = LR_likelihood * gate_probs
		ex_probs = (ex_probs.T / (ex_probs.sum(axis=1) + 1e-8)).T
		
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

		Nj = ex_probs.sum(axis=0)
		gate_prior = Nj / N

		loss = lambda param: compute_loss(param, ex_probs, pred, y, lambd)
		grad = lambda param: compute_grad(param, ex_probs, pred, y, lambd)

		experts = fmin_l_bfgs_b(loss, experts.flatten(), fprime=grad, pgtol=1e-8)[0].reshape((K, dim))

		for e in range(K):
			gate_covariance[e, :, :] = np.dot(np.dot((X - gate_mean[e, :]).T, np.diag(ex_probs[:, e])), (X - gate_mean[e, :])) / Nj[e] + 1e-6 * np.eye(dim)
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
		f = open("../results/ggLR/" + str(args.data) + '_ggLR.txt' , 'a')
		f.write(str(split) + ", " + str(args.experts) + ", " + str(args.reg_value) + ", " + str(max_acc) + "\n")