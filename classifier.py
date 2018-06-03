# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import pickle

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

from utils import load_mnist

FLAGS = None


# losses

def one_hot_encoder(data):
	data = data.astype(np.int32)
	onehot = np.zeros((len(data), 10))
	data -= 1
	onehot[np.arange(len(data)), data] = 1

	return onehot


def conv2d(x, W):
	"""conv2d returns a 2d convolution layer with full stride."""
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
	"""max_pool_2x2 downsamples a feature map by 2X."""
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
	"""weight_variable generates a weight variable of a given shape."""
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)


def bias_variable(shape):
	"""bias_variable generates a bias variable of a given shape."""
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)


def variable_summaries(var, name):
	# """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
	# with tf.name_scope(name):
	# 	mean = tf.reduce_mean(var)
	# 	tf.summary.scalar('mean', mean)
	# 	with tf.name_scope('stddev'):
	# 		stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
	# 	tf.summary.scalar('stddev', stddev)
	# 	tf.summary.scalar('max', tf.reduce_max(var))
	# 	tf.summary.scalar('min', tf.reduce_min(var))
	# 	tf.summary.histogram('histogram', var)
	pass


class CNNClassifier():
	def __init__(self, classifier_name, load_from_pkl=False, pkl_fname=None, dir=None):
		self.num_epochs = 100
		self.classifier_name = classifier_name
		self.log_dir = 'logs/{}/'.format(classifier_name)
		self.batch_size = 64
		self.dropout_prob = 0.9
		self.save_to = classifier_name + "_classifier.pkl"
		self.lamb = 1e-3
		self.c_dim = 1
		if load_from_pkl:
			self.real_mnist_x, self.real_mnist_y = load_mnist('mnist')
			self.test_labels = self.real_mnist_y
			# self.test_labels.astype(np.float32, copy=False)
			self.test_images = self.real_mnist_x.reshape(-1, 784)

			pkl_label_path = "{}edited_generated_labels_{}.pkl".format(dir, pkl_fname)
			pkl_path = "{}edited_generated_trainingset_{}.pkl".format(dir, pkl_fname)
			self.set_log_dir("{}".format(pkl_fname))
			self.data_X = pickle.load(open(pkl_path, 'rb'))
			self.data_y = pickle.load(open(pkl_label_path, 'rb'))
		if "custom" in self.classifier_name:
			self.IMAGE_WIDTH = 28
			self.IMAGE_HEIGHT = 28

		if self.classifier_name == 'mnist' or self.classifier_name == 'fashion-mnist':
			self.IMAGE_WIDTH = 28
			self.IMAGE_HEIGHT = 28
			# mnist = input_data.read_data_sets('../data/mnist', one_hot=True)
			self.data_X, self.data_y = load_mnist(self.classifier_name)

			self.test_images = self.data_X[:1000].reshape(-1, 784)
			self.test_labels = self.data_y[:1000]  # self.get_batch = mnist.train.next_batch(self.batch_size)  # self.mnist = mnist
		# elif self.classifier_name == "cifar10":
		# 	self.IMAGE_WIDTH = 32
		# 	self.IMAGE_HEIGHT = 32
		# 	self.c_dim = 3
		# 	self.data_X, self.data_y, self.test_images, self.test_labels = get_train_test_data()
		# 	self.test_images = self.test_images.reshape(-1, 1024)
		# 	# get number of batches for a single epoch
		# 	self.num_batches = len(self.data_X) // self.batch_size

		# init_variables try to load from pickle:
		try:
			self.load_model()
		except:
			# Model params
			self.W_conv1 = weight_variable([5, 5, self.c_dim, 32])
			self.b_conv1 = bias_variable([32])
			self.W_conv2 = weight_variable([5, 5, 32, 64])
			self.b_conv2 = bias_variable([64])
			self.W_fc1 = weight_variable([int(self.IMAGE_HEIGHT / 4) * int(self.IMAGE_HEIGHT / 4) * 64, 1024])
			self.b_fc1 = bias_variable([1024])
			self.W_fc2 = weight_variable([1024, 10])
			self.b_fc2 = bias_variable([10])

		self._create_model()

	def set_log_dir(self, log_dir_name):
		self.log_dir = "logs/{}".format(log_dir_name)

	def set_dataset(self, training, labels):
		training, labels = shuffle(training, labels, random_state=0)
		self.data_X = np.asarray(training[1000:]).reshape(-1, 784)
		self.data_y = np.asarray(labels[1000:]).reshape(-1, 10)
		self.test_images = self.data_X.reshape(-1, 784)
		self.test_labels = self.data_y  # self.get_batch = mnist.train.next_batch(self.batch_size)  # self.mnist = mnist

	def _deepcnn(self, x, keep_prob):
		with tf.name_scope('reshape'):
			x_image = tf.reshape(x, [-1, self.IMAGE_WIDTH, self.IMAGE_HEIGHT, self.c_dim])
		h_conv1 = tf.nn.relu(conv2d(x_image, self.W_conv1) + self.b_conv1)
		h_pool1 = max_pool_2x2(h_conv1)

		h_conv2 = tf.nn.relu(conv2d(h_pool1, self.W_conv2) + self.b_conv2)
		h_pool2 = max_pool_2x2(h_conv2)
		h_pool2_flat = tf.reshape(h_pool2, [-1, int(self.IMAGE_HEIGHT / 4) * int(self.IMAGE_HEIGHT / 4) * 64])

		h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, self.W_fc1) + self.b_fc1)

		h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
		y_conv = tf.matmul(h_fc1_drop, self.W_fc2) + self.b_fc2

		# summary
		variable_summaries(self.W_conv1, 'W_conv1')
		variable_summaries(self.W_conv2, 'W_conv2')
		variable_summaries(self.b_conv1, 'b_conv1')
		variable_summaries(self.b_conv2, 'b_conv2')
		variable_summaries(self.W_fc1, 'W_fc1')
		variable_summaries(self.W_fc2, 'W_fc2')
		variable_summaries(self.b_fc1, 'b_fc1')
		variable_summaries(self.b_fc2, 'b_fc2')
		return y_conv

	def _create_model(self):
		self.x = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT * self.IMAGE_WIDTH], name="data")
		self.y_ = tf.placeholder(tf.float32, [None, 10], name="labels")
		self.keep_prob = tf.placeholder(tf.float32, name="dropout")
		# Build the graph for the deep net
		self.y_conv = self._deepcnn(self.x, self.keep_prob)

		# loss
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv)
		self.l2_regularization = self.lamb * tf.nn.l2_loss(self.W_conv1) + self.lamb * tf.nn.l2_loss(
			self.W_conv1) + self.lamb * tf.nn.l2_loss(self.W_fc1) + self.lamb * tf.nn.l2_loss(self.W_fc2)
		cross_entropy = tf.reduce_mean(cross_entropy)
		self.cross_entropy = cross_entropy
		cross_entropy += self.l2_regularization
		tf.summary.scalar('cross_entropy', cross_entropy)

		self.train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

		correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
		correct_prediction = tf.cast(correct_prediction, tf.float32)
		self.accuracy = tf.reduce_mean(correct_prediction)
		tf.summary.scalar('accuracy', self.accuracy)

		self.confidence = tf.cast(tf.reduce_mean(tf.reduce_max(tf.nn.softmax(self.y_conv), axis=-1), axis=0), tf.float32)
		tf.summary.scalar('confidence', self.confidence)

		self.argmax = tf.argmax(self.y_conv, 1)

		graph_location = self.log_dir + 'train'
		graph_location_test = self.log_dir + 'test'
		self.merged = tf.summary.merge_all()
		print('Saving graph to: %s' % graph_location)
		self.train_writer = tf.summary.FileWriter(graph_location)
		self.test_writer = tf.summary.FileWriter(graph_location_test)
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())
		self.train_writer.add_graph(self.sess.graph)
		self.test_writer.add_graph(self.sess.graph)

	def train(self):
		start_batch_id = 0  # int(1000 / self.batch_size)
		self.num_batches = len(self.data_X) // self.batch_size
		for epoch in range(self.num_epochs):
			for i in range(start_batch_id, self.num_batches):
				batch_images = self.data_X[i * self.batch_size:(i + 1) * self.batch_size].reshape(-1, self.IMAGE_WIDTH * self.IMAGE_HEIGHT)
				batch_labels = self.data_y[i * self.batch_size:(i + 1) * self.batch_size]

				if i % 500 == 0:
					# import matplotlib.pyplot as plt
					# plt.title("label{}".format(self.data_y[0]))
					# plt.imshow(self.data_X[0].reshape(28, 28))
					# plt.show()
					# plt.title("label{}".format(self.data_y[1]))
					# plt.imshow(self.data_X[1].reshape(28, 28))
					# plt.show()
					self.test_labels, self.test_images = shuffle(self.test_labels, self.test_images, random_state=0)
					self.test(self.test_images[:1000].reshape(-1, 784), self.test_labels[:1000].reshape(-1, 10), epoch * i)
					summary, _ = self.sess.run([self.merged, self.train_step],
					                           feed_dict={self.x: batch_images, self.y_: batch_labels, self.keep_prob: 1.})
					self.train_writer.add_summary(summary, i)
					print('train accuracy epoch{}: step{}/{}'.format(epoch, i, self.num_batches))
				else:
					self.train_step.run(session=self.sess,
					                    feed_dict={self.x: batch_images, self.y_: batch_labels, self.keep_prob: self.dropout_prob})
		self.save_model()

	def test(self, test_batch, test_labels, counter=0, is_arg_max=False):
		if is_arg_max:
			summary, accuracy, confidence, loss, arg_max = self.sess.run(
				[self.merged, self.accuracy, self.confidence, self.cross_entropy, self.argmax],
				feed_dict={self.x: test_batch, self.y_: test_labels, self.keep_prob: 1.})
			print("argmax:{}".format(arg_max))
			self.test_writer.add_summary(summary, counter)
			print('step {}: accuracy:{}, confidence:{}, loss:{}'.format(counter, accuracy, confidence, loss))
			return accuracy, confidence, loss, arg_max
		else:
			summary, accuracy, confidence, loss = self.sess.run([self.merged, self.accuracy, self.confidence, self.cross_entropy],
			                                                    feed_dict={self.x: test_batch, self.y_: test_labels, self.keep_prob: 1.})
			self.test_writer.add_summary(summary, counter)
			print('step {}: accuracy:{}, confidence:{}, loss:{}'.format(counter, accuracy, confidence, loss))
			return accuracy, confidence, loss

	def save_model(self):

		# Save the model for a pickle
		pickle.dump([self.sess.run(self.W_conv1), self.sess.run(self.b_conv1), self.sess.run(self.W_conv2), self.sess.run(self.b_conv2),
		             self.sess.run(self.W_fc1), self.sess.run(self.b_fc1), self.sess.run(self.W_fc2), self.sess.run(self.b_fc2)],
		            open(self.save_to, 'wb'))

		print("Model has been saved!")

	def load_model(self):
		model = pickle.load(open(self.save_to, 'rb'))
		self.W_conv1 = tf.Variable(tf.constant(model[0]))
		self.b_conv1 = tf.Variable(tf.constant(model[1]))
		self.W_conv2 = tf.Variable(tf.constant(model[2]))
		self.b_conv2 = tf.Variable(tf.constant(model[3]))
		self.W_fc1 = tf.Variable(tf.constant(model[4]))
		self.b_fc1 = tf.Variable(tf.constant(model[5]))
		self.W_fc2 = tf.Variable(tf.constant(model[6]))
		self.b_fc2 = tf.Variable(tf.constant(model[7]))
		print("model has been loaded from {}".format(self.save_to))


def parse_args():
	desc = "Tensorflow implementation of GAN collections"
	parser = argparse.ArgumentParser(description=desc)
	parser.add_argument('--dir_name', type=str, default='')
	parser.add_argument('--preprocess', type=bool, default=False)
	parser.add_argument('--fname', type=str, default='fashion-mnist_MultivariateGaussianSampler')

	return parser.parse_args()


def preprocess_data(dir, pkl_fname, batch_size=64):
	# mapping only once need to edit the condition
	pkl_label_path = "{}generated_labels_{}.pkl".format(dir, pkl_fname)
	pkl_path = "{}generated_trainingset_{}.pkl".format(dir, pkl_fname)
	data_X = pickle.load(open(pkl_path, 'rb'))
	data_y = pickle.load(open(pkl_label_path, 'rb'))

	data_X = np.asarray([y for x in data_X for y in x]).reshape(-1, 28, 28)

	data_y = np.asarray(data_y, dtype=np.float32)
	data_y_categorical = data_y
	data_y = one_hot_encoder(data_y)
	pretraind = CNNClassifier('mnist')
	indices = np.argwhere(data_y == 1)
	for i in range(10):
		mask = (indices[:, 1] == i)
		tmp = data_X[np.where(mask == True)][:60]
		dummy_labels = np.repeat(np.arange(10), 6)
		# to one hot vec
		z = np.zeros((60, 10))
		for j, l in enumerate(dummy_labels):
			z[j, l] = 1
		dummy_labels = z
		# import matplotlib.pyplot as plt

		# plt.imshow(tmp[0].reshape(28, 28))
		# plt.show()
		# plt.imshow(tmp[1].reshape(28, 28))
		# plt.show()
		# data_y = one_hot_encoder(data_y)
		_, _, _, arg_max = pretraind.test(tmp.reshape(-1, 784), dummy_labels.reshape(-1, 10), is_arg_max=True)
		data_y_categorical[mask] = np.bincount(arg_max).argmax() + 1
		print(np.bincount(arg_max))

	data_y = one_hot_encoder(data_y_categorical)
	data_X, data_y = shuffle(data_X, data_y, random_state=0)
	pickle.dump(data_y, open("{}edited_generated_labels_{}.pkl".format(dir, pkl_fname), 'wb'))
	pickle.dump(data_X, open("{}edited_generated_trainingset_{}.pkl".format(dir, pkl_fname), 'wb'))
	del pretraind
	del data_y
	del data_X


def main():
	# parse arguments
	args = parse_args()
	if args is None:
		exit()
	fname = args.fname
	dir = args.dir_name
	do_preprocess = args.preprocess
	if do_preprocess:
		preprocess_data(dir, fname)

	else:
		c = CNNClassifier("custom", load_from_pkl=True, pkl_fname=fname, dir=dir)
		c.train()


if __name__ == '__main__':
	# c = CNNClassifier("custom", load_from_pkl=True,
	#                   pkl_path="/Users/idan.a/results_22_5/generated_trainingset_fashion-mnist_MultivariateGaussianSampler.pkl",
	#                   pkl_label_path="/Users/idan.a/results_22_5/generated_labels_fashion-mnist_MultivariateGaussianSampler.pkl")
	# c.train()
	main()
