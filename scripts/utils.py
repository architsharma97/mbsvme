import numpy as np
import scipy.io as spo
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import OneHotEncoder

def preprocessing(X, preprocess='gauss', return_stats=False):
	# basic preprocessing
	X = np.array(X)
	if preprocess == 'gauss':
		m, std = X.mean(axis=0),  X.std(axis=0)
		std += 1e-8
	elif preprocess == 'standard':
		m, std = X.min(axis=0), X.max(axis=0)
		std = std - m
		std += 1e-8
	else:
		m, std = 0., 1.0
	
	# the first feature is for the bias, which would std = 0, mean = 1
	if preprocess != 'none':
		m[0], std[0] = 0., 1.0
	
	if return_stats:
		return (X - m) / std, m, std
	else:
		return (X - m) / std

def read_data(key='synth', return_split=False, preprocess='gauss', split_id=-1):
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

	if key in ['pima', 'wisconsin', 'sonar']:
		if key in ['pima', 'wisconsin']:
			data = np.genfromtxt( "../data/" + key + "/data.csv", delimiter=',')
		if key in ['sonar']:
			data = np.genfromtxt( "../data/" + key + "/data.csv", delimiter=' ')
		
		data = np.concatenate((np.ones((data.shape[0], 1)), data), axis=1)

		np.random.shuffle(data)
		data[:, -1] = 2 * (data[:, -1] == 1) - 1
		return data, None, None, None
	
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
		if split_id == -1:
			if key in ['image', 'splice']:
				split_id = np.random.randint(0, 20)
			else:
				split_id = np.random.randint(0,100)

		split = split_id
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
	
	if key == 'adult':
		data = np.genfromtxt("../data/adult/data.csv", delimiter=',')
		X = data[:22696, 1:]
		y = data[:22696, 0]
		Xt = data[22696:, 1:]
		yt = data[22696:, 0]
		
		# adding bias terms
		X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
		Xt = np.concatenate((np.ones((Xt.shape[0], 1)), Xt), axis=1)

		# categorical data, will skip preprocessing
		return X, y, Xt, yt

	if key == 'parkinsons':
		data = np.genfromtxt("../data/parkinsons/data.csv", delimiter=',')
		np.random.shuffle(data)
		# labels at the end
		data = np.concatenate((np.ones((data.shape[0], 1)), data[:, 1:], data[:, 0:1]), axis=1)

		# make labels 1, -1
		data[:, -1] = 2 * (data[:, -1] == 1) - 1
		return data, None, None, None

	if key in ['landmine', 'mnist', 'sentiment']:
		if key == 'landmine':
			data = spo.loadmat('../data/landmine_balanced.mat')
			tasks = 19
		elif key == 'mnist':
			data = spo.loadmat('../data/mnist.mat')
			tasks = 10
		elif key == 'sentiment':
			data = spo.loadmat('../data/sentiment_processed.mat')
			tasks = 4
		
		stats = [preprocessing(np.concatenate([np.ones((data['xTr'][0, idx].shape[0], 1)), data['xTr'][0, idx]], axis=1), preprocess=preprocess, return_stats=True) for idx in range(tasks)]
		
		xTr = [stats[idx][0] for idx in range(tasks)]
		xTe = [(np.concatenate([np.ones((data['xTe'][0, idx].shape[0], 1)), data['xTe'][0, idx]], axis=1) - stats[idx][1]) / stats[idx][2] for idx in range(tasks)]
		
		if key == 'sentiment':
			yTr = [2 * data['yTr'][0, idx][:, 0].astype('int8') - 3 for idx in range(tasks)]
			yTe = [2 * data['yTe'][0, idx][:, 0].astype('int8') - 3 for idx in range(tasks)]
		else:
			yTr = [data['yTr'][0, idx][:, 0] for idx in range(tasks)]
			yTe = [data['yTe'][0, idx][:, 0] for idx in range(tasks)]
		
		return xTr, yTr, xTe, yTe

	if key in ['landmine_c', 'mnist_c', 'sentiment_c']:
		if key == 'landmine_c':
			data = spo.loadmat('../data/landmine_balanced.mat')
			tasks = 19
		elif key == 'mnist_c':
			data = spo.loadmat('../data/mnist.mat')
			tasks = 10
		elif key == 'sentiment_c':
			data = spo.loadmat('../data/sentiment_processed.mat')
			tasks = 4

		X = np.concatenate([np.concatenate([np.ones((data['xTr'][0, idx].shape[0], 1)), data['xTr'][0, idx], idx * np.ones((data['xTr'][0, idx].shape[0], 1))], axis=1) for idx in range(tasks)], axis=0)
		Xt = np.concatenate([np.concatenate([np.ones((data['xTe'][0, idx].shape[0], 1)), data['xTe'][0, idx], idx * np.ones((data['xTe'][0, idx].shape[0], 1))], axis=1) for idx in range(tasks)], axis=0)
		
		if key == 'sentiment_c':
			y = np.concatenate([2 * data['yTr'][0, idx][:, 0].astype('int8') - 3 for idx in range(tasks)], axis=0)
			yt = np.concatenate([2 * data['yTe'][0, idx][:, 0].astype('int8') - 3 for idx in range(tasks)], axis=0)
		else:
			y = np.concatenate([data['yTr'][0, idx][:, 0] for idx in range(tasks)], axis=0)
			yt = np.concatenate([data['yTe'][0, idx][:, 0] for idx in range(tasks)], axis=0)
	
	X, m, std = preprocessing(X, preprocess=preprocess, return_stats=True)

	# return train, train labels, test, test labels, and split if enabled
	if return_split == False:
		return X, np.array(y), (np.array(Xt) - m) / std, np.array(yt)
	else:
		return X, np.array(y), (np.array(Xt) - m) / std, np.array(yt), split

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

def process_results(fname, hypercount=3, splits=10, val=False):
	f = open(fname, 'r').read().splitlines()
	acc = {}

	for line in f:
		tokens = line.split(',')
		line_id = tuple([float(tokens[idx]) for idx in range(1, hypercount+1)]) 
		if line_id not in acc.keys():
			acc[line_id] = [[] for _ in range(splits)]

		acc[line_id][int(tokens[0])].append(float(tokens[-1]))

	best_config = None
	for key in acc.keys():
		data = np.array(acc[key])
		if len(data.shape) < 2:
			continue
		if val:
			std = np.array(acc[key]).std(axis=0)
			idx = np.argmax(np.array(acc[key]).mean(axis=0))
			acc[key], std = np.max(np.array(acc[key]).mean(axis=0)), std[idx]
		else:
			std = np.max(np.array(acc[key]), axis=1).std()
			acc[key] = np.max(np.array(acc[key]), axis=1).mean()

		if best_config is None or best_config[0] < acc[key]:
			best_config = (acc[key], std, key)

	print best_config

def process_results_robust(fname, hypercount=2):
	f = open(fname, 'r').read().splitlines()
	keys = list(set([tuple([float(x.split(',')[idx]) for idx in range(1, hypercount + 1)]) for x in f]))

	acc = {}
	for key in keys:
		acc[key] = []

	for line in f:
		acc[tuple([float(line.split(',')[idx]) for idx in range(1, hypercount + 1)])].append(float(line.split(',')[-1]))

	for key in acc.keys():
		print 100-np.mean(acc[key]), np.std(acc[key]), len(acc[key])