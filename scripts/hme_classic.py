import sys
sys.path.append('classicHME/scripts/')

import numpy as np
from numpy import linalg as LA
from scipy.stats import multivariate_normal

from utils import read_data
import general_hme as hm

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

hme = hm.HME(y, X, yt, Xt, "softmax", bias=False, gate_type="softmax", verbose=False, levels=args.levels, branching=2)
hme.fit()
hme_predict = hme.predict(Xt)

print np.mean(hme_predict == yt) * 100