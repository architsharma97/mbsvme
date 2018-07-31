import numpy as np
from utils import read_data
import tensorflow as tf
import argparse

# ---------------------------------------------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=str, default='banana',
					help='Name of the dataset')
args = parser.parse_args()
# ---------------------------------------------------------------------------------------------------------------------------------------------------------

X, y, Xt, yt = read_data(key=args.data)

# labels should 0-1
y = ((y + 1)/2).reshape(len(y), 1)
yt = ((yt + 1)/2).reshape(len(yt), 1)

dim = X.shape[1]

inp = tf.placeholder(tf.float32, shape=[None, dim])
labels = tf.placeholder(tf.float32, shape=[None, 1])

# params
hsize = 1024
W1 = tf.get_variable("W1", shape=[dim, hsize], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.get_variable("b1", shape=[hsize], initializer=tf.zeros_initializer())

W2 = tf.get_variable("W2", shape=[hsize, hsize], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.get_variable("b2", shape=[hsize], initializer=tf.zeros_initializer())

W3 = tf.get_variable("W3", shape=[hsize, hsize], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.get_variable("b3", shape=[hsize], initializer=tf.zeros_initializer())

W4 = tf.get_variable("W4", shape=[hsize, 1], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.get_variable("b4", shape=[1], initializer=tf.zeros_initializer())

# graph
h1 = tf.nn.relu(tf.matmul(inp, W1) + b1)
# h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
h3 = tf.nn.relu(tf.matmul(h1, W3) + b3)
pred = tf.matmul(h3, W4) + b4

# training
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=pred))
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(labels, 'int32'), tf.cast(pred > 0.0, 'int32')),'float32'))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
max_acc = -1
for i in range(50):
	tr_loss, _ = sess.run([loss, train_step], feed_dict={inp: X, labels: y})
	test_acc = sess.run([accuracy], feed_dict={inp : Xt, labels: yt})
	max_acc = max(max_acc, test_acc[0])
	print("Test Accuracy: " + str(test_acc[0]))

print("Maximum Test Accuracy: " + str(max_acc*100))