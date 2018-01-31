import numpy as np
import scipy.io as spo
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import OneHotEncoder

def read_data(key='synth', return_split=False):
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

	if key in ['banana', 'breast_cancer', 'diabetis', 'flare_solar', 'german', 'heart', 'image', 'ringnorm', 'splice', 'titanic', 'waveform']:
		data = spo.loadmat('../data/benchmarks.mat')[key]
		
		# image and splice have 20 splits only, rest have 100 splits
		if key in ['image', 'splice']:
			rand_idx = np.random.randint(0, 20)
		else:
			rand_idx = np.random.randint(0,100)

		split = rand_idx
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

	if key == 'ijcnn':
		train = open('../data/ijcnn/train.csv','r').read().splitlines()
		val = open('../data/ijcnn/val.csv', 'r').read().splitlines()
		test = open('../data/ijcnn/test.csv', 'r').read().splitlines()

		X, y  = [], []
		for line in train:
			X.append([1.0] + [float(f) for f in line.split(',')[1:]])
			y.append(int(float(line.split(',')[0])))

		# split into test and val later
		Xt, yt = [], []
		for line in val:
			Xt.append([1.0] + [float(f) for f in line.split(',')[1:]])
			yt.append(int(float(line.split(',')[0])))

		for line in test:
			Xt.append([1.0] + [float(f) for f in line.split(',')[1:]])
			yt.append(int(float(line.split(',')[0])))
		

	# basic preprocessing
	X = np.array(X)
	m, std = X.mean(axis=0),  X.std(axis=0)

	# the first feature is for the bias, which would std = 0, mean = 1
	m[0], std[0] = 0., 1.0

	# return train, train labels, test, test labels, and split if enabled
	if return_split == False:
		return (X - m) / std, np.array(y), (np.array(Xt) - m) / std, np.array(yt)
	else:
		return (X - m) / std, np.array(y), (np.array(Xt) - m) / std, np.array(yt), split

def softmaxx(X):
	cur = np.exp(X.T - np.max(X, axis=1)).T
	return (cur.T / cur.sum(axis=1) + 1e-6).T

def init(rows, cols, key='gauss', data=None):
	if key == 'gauss':
		return np.random.randn(rows, cols)

	elif key == 'kmeans':
		kmeans = KMeans(n_clusters=rows).fit(data)
		cluster_ids = kmeans.predict(data)
		labels = OneHotEncoder(sparse=False).fit_transform(cluster_ids.reshape(len(cluster_ids), 1))
		
		weights = np.random.randn(rows, cols)
		N = data.shape[0]

		for _ in range(50):
			pred = softmaxx(np.dot(data, weights.T))
			weights += 0.1 * np.dot((labels - pred).T, data) / N

		print "Accuracy of initialization: " + str(np.mean((np.argmax(softmaxx(np.dot(data, weights.T)), axis=1) == cluster_ids)) * 100)
		return weights
