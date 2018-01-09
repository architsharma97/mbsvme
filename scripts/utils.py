import numpy as np
import scipy.io as spo

def read_data(key='synth'):
	if key == 'synth':
		train = open('../data/synth.tr','r').read().splitlines()[1:]
		test = open('../data/synth.te', 'r').read().splitlines()[1:]
		X, y, Xt, yt = [], [], [], []

		for entry in train:
			entries = entry.split(' ')
			# -1, 1
			y.append(2 * int(entries[-1]) - 1)
			X.append([1.0] + [float(t) for t in entries[:-1] if t != ''])

		for entry in test:
			entries = entry.split(' ')
			yt.append(2 * int(entries[-1]) - 1)
			Xt.append([1.0] + [float(t) for t in entries[:-1] if t != ''])
	
	if key == 'pima':
		train = open('../data/pima.tr', 'r').read().splitlines()[1:]
		test = open('../data/pima.te', 'r').read().splitlines()[1:]

		X, y, Xt, yt = [], [], [], []

		for entry in train:
			entries = entry.split(' ')
			# -1, 1
			y.append(2 * (entries[-1] == 'Yes') - 1)
			X.append([1.0] + [float(t) for t in entries[:-1] if t != ''])

		for entry in test:
			entries = entry.split(' ')
			yt.append(2 * (entries[-1] == 'Yes') - 1)
			Xt.append([1.0] + [float(t) for t in entries[:-1] if t != ''])

	if key in ['banana', 'breast_cancer', 'diabetis', 'flare_solar', 'german', 'heart', 'image', 'ringnorm', 'splice', 'image', 'titanic', 'waveform']:
		data = spo.loadmat('../data/benchmarks.mat')[key]
		rand_idx = np.random.randint(0, 100)

		# temporarily restrict to one split
		split = 42
		train_indices = data['train'][0][0][split, :]
		test_indices = data['test'][0][0][split, :]

		X = np.array(data['x'][0][0][train_indices - 1, :])
		y = np.array(data['t'][0][0][train_indices - 1, :])
		Xt = np.array(data['x'][0][0][test_indices - 1, :])
		yt = np.array(data['t'][0][0][test_indices - 1, :])

		X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
		Xt = np.concatenate((np.ones((Xt.shape[0], 1)), Xt), axis=1)
		y = y.sum(axis=1)
		yt = yt.sum(axis=1)

	# basic preprocessing
	X = np.array(X)
	m, std = X.mean(axis=0),  X.std(axis=0)

	# the first feature is for the bias, which would std = 0, mean = 1
	m[0], std[0] = 0., 1.0

	# return train, train labels, test, test labels
	return (X - m) / std, np.array(y), (np.array(Xt) - m) / std, np.array(yt)