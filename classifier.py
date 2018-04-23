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

import pickle

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None


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
	"""Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
	with tf.name_scope(name):
		mean = tf.reduce_mean(var)
		tf.summary.scalar('mean', mean)
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
		tf.summary.scalar('stddev', stddev)
		tf.summary.scalar('max', tf.reduce_max(var))
		tf.summary.scalar('min', tf.reduce_min(var))
		tf.summary.histogram('histogram', var)


class CNNClassifier():
	def __init__(self, classifier_name):
		self.classifier_name = classifier_name
		self.log_dir = 'logs/mnist'
		self.batch_size = 64
		self.dropout_prob = 0.9
		self.save_to = classifier_name + "_classifier.pkl"
		if self.classifier_name == 'mnist':
			mnist = input_data.read_data_sets('../data/mnist', one_hot=True)
			self.test_images = mnist.test.images
			self.test_labels = mnist.test.labels
			self.get_batch = mnist.train.next_batch(self.batch_size)

		# init_variables try to load from pickle:
		try:
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
		except:
			# Model params
			self.W_conv1 = weight_variable([5, 5, 1, 32])
			self.b_conv1 = bias_variable([32])
			self.W_conv2 = weight_variable([5, 5, 32, 64])
			self.b_conv2 = bias_variable([64])
			self.W_fc1 = weight_variable([7 * 7 * 64, 1024])
			self.b_fc1 = bias_variable([1024])
			self.W_fc2 = weight_variable([1024, 10])
			self.b_fc2 = bias_variable([10])
		self._create_model()

	def _deepcnn(self, x, keep_prob):
		with tf.name_scope('reshape'):
			x_image = tf.reshape(x, [-1, 28, 28, 1])
		h_conv1 = tf.nn.relu(conv2d(x_image, self.W_conv1) + self.b_conv1)
		h_pool1 = max_pool_2x2(h_conv1)

		h_conv2 = tf.nn.relu(conv2d(h_pool1, self.W_conv2) + self.b_conv2)
		h_pool2 = max_pool_2x2(h_conv2)
		h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

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
		self.x = tf.placeholder(tf.float32, [None, 784])
		self.y_ = tf.placeholder(tf.float32, [None, 10])
		self.keep_prob = tf.placeholder(tf.float32)
		# Build the graph for the deep net
		self.y_conv = self._deepcnn(self.x, self.keep_prob)

		# loss
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv)
		cross_entropy = tf.reduce_mean(cross_entropy)
		tf.summary.scalar('cross_entropy', cross_entropy)

		self.train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

		correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
		correct_prediction = tf.cast(correct_prediction, tf.float32)
		self.accuracy = tf.reduce_mean(correct_prediction)
		tf.summary.scalar('accuracy', self.accuracy)

		confidence = tf.cast(tf.reduce_max(self.y_conv, 1), tf.float32)
		self.confidence = tf.reduce_mean(tf.nn.softmax(confidence))
		tf.summary.scalar('confidence', self.confidence)

		graph_location = self.log_dir + '/train'
		graph_location_test = self.log_dir + '/test'
		self.merged = tf.summary.merge_all()
		print('Saving graph to: %s' % graph_location)
		self.train_writer = tf.summary.FileWriter(graph_location)
		self.test_writer = tf.summary.FileWriter(graph_location_test)

	def train(self):
		self.sess = tf.Session()
		self.train_writer.add_graph(self.sess.graph)
		self.sess.run(tf.global_variables_initializer())

		for i in range(10000):
			batch = self.get_batch

			if i % 300 == 0:
				self.test(self.test_images, self.test_labels, i)
				summary, _ = self.sess.run([self.merged, self.train_step],
				                           feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 1})
				self.train_writer.add_summary(summary, i)
				print('train accuracy step {}'.format(i))
			else:
				self.train_step.run(session=self.sess, feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: self.dropout_prob})
		self.save_model()

	def test(self, test_batch, test_labels, counter):
		summary, accuracy, confidence = self.sess.run([self.merged, self.accuracy, self.confidence],
		                              feed_dict={self.x: test_batch, self.y_: test_labels, self.keep_prob: 1})

		print('step {}: accuracy:{}, confidence:{}'.format(counter,accuracy, confidence))
		self.test_writer.add_summary(summary, counter)

	def save_model(self):

		# Save the model for a pickle
		pickle.dump([self.sess.run(self.W_conv1), self.sess.run(self.b_conv1), self.sess.run(self.W_conv2), self.sess.run(self.b_conv2),
		             self.sess.run(self.W_fc1), self.sess.run(self.b_fc1), self.sess.run(self.W_fc2), self.sess.run(self.b_fc2)],
		            open(self.save_to, 'wb'))

	print("Model has been saved!")


if __name__ == '__main__':
	c = CNNClassifier("mnist")
	c.train()