# -*- coding: utf-8 -*-
from __future__ import division

import time

from matplotlib.legend_handler import HandlerLine2D
from sklearn.manifold import TSNE

from Sampler import *
from cifar10 import *
from classifier import CNNClassifier
from ops import *
from utils import *


def conv2d_(input, name, kshape, strides=[1, 1, 1, 1]):
	with tf.name_scope(name):
		W = tf.get_variable(name='w_' + name, shape=kshape, initializer=tf.contrib.layers.xavier_initializer(uniform=False))
		b = tf.get_variable(name='b_' + name, shape=[kshape[3]], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
		out = tf.nn.conv2d(input, W, strides=strides, padding='SAME')
		out = tf.nn.bias_add(out, b)
		out = tf.nn.relu(out)
		return out


# ---------------------------------
def deconv2d_(input, name, kshape, n_outputs, strides=[1, 1]):
	with tf.name_scope(name):
		out = tf.contrib.layers.conv2d_transpose(input, num_outputs=n_outputs, kernel_size=kshape, stride=strides, padding='SAME',
		                                         weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False),
		                                         biases_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
		                                         activation_fn=tf.nn.relu)
		return out


#   ---------------------------------
def maxpool2d(x, name, kshape=[1, 2, 2, 1], strides=[1, 2, 2, 1]):
	with tf.name_scope(name):
		out = tf.nn.max_pool(x, ksize=kshape,  # size of window
		                     strides=strides, padding='SAME')
		return out


#   ---------------------------------
def upsample(input, name, factor=[2, 2]):
	size = [int(input.shape[1] * factor[0]), int(input.shape[2] * factor[1])]
	with tf.name_scope(name):
		out = tf.image.resize_bilinear(input, size=size, align_corners=None, name=None)
		return out


#   ---------------------------------
def fullyConnected(input, name, output_size):
	with tf.name_scope(name):
		input_size = input.shape[1:]
		input_size = int(np.prod(input_size))
		W = tf.get_variable(name='w_' + name, shape=[input_size, output_size],
		                    initializer=tf.contrib.layers.xavier_initializer(uniform=False))
		b = tf.get_variable(name='b_' + name, shape=[output_size], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
		input = tf.reshape(input, [-1, input_size])
		out = tf.nn.relu(tf.add(tf.matmul(input, W), b))
		return out


#   ---------------------------------
def dropout(input, name, keep_rate):
	with tf.name_scope(name):
		out = tf.nn.dropout(input, keep_rate)
		return out

def tsne(X, k=2, perplexity=100):
	tsne = TSNE(n_components=k, init='pca', random_state=0, perplexity=perplexity)
	X_transformed = tsne.fit_transform(X)
	# pca = RandomizedPCA(n_components=2)
	# X_transformed = pca.fit_transform(X)

	return X_transformed


def plot_with_images(X, images, title="", image_num=25):
	'''
	A plot function for viewing images in their embedded locations. The
	function receives the embedding (X) and the original images (images) and
	plots the images along with the embeddings.

	:param X: Nxd embedding matrix (after dimensionality reduction).
	:param images: NxD original data matrix of images.
	:param title: The title of the plot.
	:param num_to_plot: Number of images to plot along with the scatter plot.
	:return: the figure object.
	'''

	n, pixels = np.shape(images)
	img_size = int(pixels ** 0.5)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title(title)

	# get the size of the embedded images for plotting:
	x_size = (max(X[:, 0]) - min(X[:, 0])) * 0.08
	y_size = (max(X[:, 1]) - min(X[:, 1])) * 0.08

	# draw random images and plot them in their relevant place:
	for i in range(image_num):
		img_num = np.random.choice(n)
		x0, y0 = X[img_num, 0] - x_size / 2., X[img_num, 1] - y_size / 2.
		x1, y1 = X[img_num, 0] + x_size / 2., X[img_num, 1] + y_size / 2.
		img = images[img_num, :].reshape(img_size, img_size)
		ax.imshow(img, aspect='auto', cmap=plt.cm.gray, zorder=100000,
		          extent=(x0, x1, y0, y1))

	# draw the scatter plot of the embedded data points:
	ax.scatter(X[:, 0], X[:, 1], marker='.', alpha=0.7)
	plt.savefig("autoencoder_mnist_dim100.jpeg")
	plt.show()
	return fig
#   ---------------------------------
class AEMultiModalInfoGAN(object):
	model_name = "AEMultiModalInfoGAN"  # name for checkpoint

	def __init__(self, sess, epoch, batch_size, z_dim, dataset_name, checkpoint_dir, result_dir, log_dir, sampler, SUPERVISED=True):
		self.confidence_list = []
		self.sess = sess
		self.dataset_name = dataset_name
		self.checkpoint_dir = checkpoint_dir
		self.result_dir = result_dir
		self.log_dir = log_dir
		self.epoch = epoch
		self.batch_size = batch_size
		self.sampler = sampler
		self.pretrained_classifier = CNNClassifier("mnist")

		self.SUPERVISED = SUPERVISED  # if it is true, label info is directly used for code

		# train
		self.learning_rate = 0.0002
		self.beta1 = 0.5

		# test
		self.sample_num = 64  # number of generated images to be saved

		# code
		self.len_discrete_code = 10  # categorical distribution (i.e. label)
		self.len_continuous_code = 2  # gaussian distribution (e.g. rotation, thickness)
		self.embedding_size = z_dim

		if dataset_name == 'mnist' or dataset_name == 'fashion-mnist':
			# parameters
			self.input_height = 28
			self.input_width = 28
			self.output_height = 28
			self.output_width = 28

			self.z_dim = z_dim  # dimension of noise-vector
			self.y_dim = 12  # dimension of code-vector (label+two features)
			self.c_dim = 1

			# load mnist
			self.data_X, self.data_y = load_mnist(self.dataset_name)


			# get number of batches for a single epoch
			self.num_batches = len(self.data_X) // self.batch_size
		elif dataset_name == 'cifar10':
			# parameters
			self.input_height = 32
			self.input_width = 32
			self.output_height = 32
			self.output_width = 32

			self.z_dim = z_dim  # dimension of noise-vector
			self.y_dim = 12  # dimension of code-vector (label+two features)
			self.c_dim = 3
			self.data_X, self.data_y, self.test_x ,self.test_labels = get_train_test_data()

			# get number of batches for a single epoch
			self.num_batches = len(self.data_X) // self.batch_size

	def ConvAutoEncoder(self):
		with tf.name_scope("ConvAutoEncoder"):
			"""
			We want to get dimensionality reduction of 784 to 196
			Layers:
				input --> 28, 28 (784)
				conv1 --> kernel size: (5,5), n_filters:25 ???make it small so that it runs fast
				pool1 --> 14, 14, 25
				dropout1 --> keeprate 0.8
				reshape --> 14*14*25
				FC1 --> 14*14*25, 14*14*5
				dropout2 --> keeprate 0.8
				FC2 --> 14*14*5, 196 --> output is the encoder vars
				FC3 --> 196, 14*14*5
				dropout3 --> keeprate 0.8
				FC4 --> 14*14*5,14*14*25
				dropout4 --> keeprate 0.8
				reshape --> 14, 14, 25
				deconv1 --> kernel size:(5,5,25), n_filters: 25
				upsample1 --> 28, 28, 25
				FullyConnected (outputlayer) -->  28* 28* 25, 28 * 28
				reshape --> 28*28
			"""
			input = tf.reshape(self.x, shape=[-1, 28, 28, 1])
			x = tf.reshape(self.x, shape=[-1, 784])
			# coding part
			c1 = conv2d_(input, name='c1', kshape=[5, 5, 1, 25])
			p1 = maxpool2d(c1, name='p1')
			do1 = dropout(p1, name='do1', keep_rate=0.75)
			do1 = tf.reshape(do1, shape=[-1, 14 * 14 * 25])
			fc1 = fullyConnected(do1, name='fc1', output_size=14 * 14 * 5)
			do2 = dropout(fc1, name='do2', keep_rate=0.75)
			embedding = fullyConnected(do2, name='fc2', output_size=self.embedding_size)
			# Decoding part
			fc3 = fullyConnected(embedding, name='fc3', output_size=14 * 14 * 5)
			do3 = dropout(fc3, name='do3', keep_rate=0.75)
			fc4 = fullyConnected(do3, name='fc4', output_size=14 * 14 * 25)
			do4 = dropout(fc4, name='do3', keep_rate=0.75)
			do4 = tf.reshape(do4, shape=[-1, 14, 14, 25])
			dc1 = deconv2d_(do4, name='dc1', kshape=[5, 5], n_outputs=25)
			up1 = upsample(dc1, name='up1', factor=[2, 2])
			output = fullyConnected(up1, name='output', output_size=28 * 28)
			with tf.name_scope('cost'):
				cost = tf.reduce_mean(tf.square(tf.subtract(output, x)))
			return output, cost, embedding

	def classifier(self, x, is_training=True, reuse=False):
		# Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
		# Architecture : (64)5c2s-(128)5c2s_BL-FC1024_BL-FC128_BL-FC12S’
		# All layers except the last two layers are shared by discriminator
		# Number of nodes in the last layer is reduced by half. It gives better results.
		with tf.variable_scope("classifier", reuse=reuse):
			net = lrelu(bn(linear(x, 64, scope='c_fc1'), is_training=is_training, scope='c_bn1'))
			out_logit = linear(net, self.y_dim, scope='c_fc2')
			out = tf.nn.softmax(out_logit)

			return out, out_logit

	def discriminator(self, x, is_training=True, reuse=False):
		# Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
		# Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
		with tf.variable_scope("discriminator", reuse=reuse):
			net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='d_conv1'))
			net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='d_conv2'), is_training=is_training, scope='d_bn2'))
			net = tf.reshape(net, [self.batch_size, -1])
			net = lrelu(bn(linear(net, 1024, scope='d_fc3'), is_training=is_training, scope='d_bn3'))
			out_logit = linear(net, 1, scope='d_fc4')
			out = tf.nn.sigmoid(out_logit)

			return out, out_logit, net

	def generator(self, z, y, is_training=True, reuse=False):
		# Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
		# Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
		with tf.variable_scope("generator", reuse=reuse):
			# merge noise and code
			z = concat([z, y], 1)

			net = lrelu(bn(linear(z, 1024, scope='g_fc1'), is_training=is_training, scope='g_bn1'))
			net = lrelu(
				bn(linear(net, 128 * self.input_height / 4 * self.input_width / 4, scope='g_fc2'), is_training=is_training, scope='g_bn2'))
			net = tf.reshape(net, [self.batch_size, int(self.input_height / 4), int(self.input_width / 4), 128])
			net = lrelu(
				bn(deconv2d(net, [self.batch_size, int(self.input_height / 2), int(self.input_width / 2), 64], 4, 4, 2, 2, name='g_dc3'),
				   is_training=is_training, scope='g_bn3'))

			out = tf.nn.sigmoid(deconv2d(net, [self.batch_size, self.input_height, self.input_width, self.c_dim], 4, 4, 2, 2, name='g_dc4'))
			# out = tf.reshape(out, ztf.stack([self.batch_size, 784]))

			return out

	def build_model(self):
		# some parameters
		image_dims = [self.input_height, self.input_width, self.c_dim]
		bs = self.batch_size

		""" Graph Input """
		# images
		self.x = tf.placeholder(tf.float32, [bs] + image_dims, name='real_images')

		# labels
		self.y = tf.placeholder(tf.float32, [bs, self.y_dim], name='y')

		# noises
		self.z = tf.placeholder(tf.float32, [bs, self.z_dim], name='z')

		""" Loss Function """
		## 0. AE
		prediction, ae_loss, embedding = self.ConvAutoEncoder()
		self.embedding = embedding
		self.ae_loss = ae_loss

		## 1. GAN Loss
		# output of D for real images
		D_real, D_real_logits, _ = self.discriminator(self.x, is_training=True, reuse=False)

		# output of D for fake images
		self.x_ = self.generator(self.z, self.y, is_training=True, reuse=False)
		D_fake, D_fake_logits, input4classifier_fake = self.discriminator(self.x_, is_training=True, reuse=True)

		# get loss for discriminator
		d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real)))
		d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake)))

		self.d_loss = d_loss_real + d_loss_fake

		# get loss for generator
		self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake)))

		## 2. Information Loss
		code_fake, code_logit_fake = self.classifier(input4classifier_fake, is_training=True, reuse=False)
		# discrete code : categorical
		disc_code_est = code_logit_fake[:, :self.len_discrete_code]
		disc_code_tg = self.y[:, :self.len_discrete_code]
		q_disc_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=disc_code_est, labels=disc_code_tg))

		# continuous code : gaussian
		cont_code_est = code_logit_fake[:, self.len_discrete_code:]
		cont_code_tg = self.y[:, self.len_discrete_code:]
		q_cont_loss = tf.reduce_mean(tf.reduce_sum(tf.square(cont_code_tg - cont_code_est), axis=1))

		# get information loss = P(x|c)
		self.q_loss = q_disc_loss + q_cont_loss

		""" Training """
		# divide trainable variables into a group for D and a group for G
		t_vars = tf.trainable_variables()
		d_vars = [var for var in t_vars if 'd_' in var.name]
		g_vars = [var for var in t_vars if 'g_' in var.name]
		q_vars = [var for var in t_vars if ('d_' in var.name) or ('c_' in var.name) or ('g_' in var.name)]

		# optimizers
		with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
			self.ae_optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.ae_loss)
			self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(self.d_loss, var_list=d_vars)
			self.g_optim = tf.train.AdamOptimizer(self.learning_rate * 5, beta1=self.beta1).minimize(self.g_loss, var_list=g_vars)
			self.q_optim = tf.train.AdamOptimizer(self.learning_rate * 5, beta1=self.beta1).minimize(self.q_loss, var_list=q_vars)

		"""" Testing """
		# for test
		self.fake_images = self.generator(self.z, self.y, is_training=False, reuse=True)
		""" Summary """
		ae_loss_sum = tf.summary.scalar("ae_loss", ae_loss)

		d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
		d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
		d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
		g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)

		q_loss_sum = tf.summary.scalar("g_loss", self.q_loss)
		q_disc_sum = tf.summary.scalar("q_disc_loss", q_disc_loss)
		q_cont_sum = tf.summary.scalar("q_cont_loss", q_cont_loss)

		# final summary operations
		self.ae_sum = tf.summary.merge([ae_loss_sum])
		self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
		self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])
		self.q_sum = tf.summary.merge([q_loss_sum, q_disc_sum, q_cont_sum])

	def train(self):

		# initialize all variables
		tf.global_variables_initializer().run()

		# graph inputs for visualize training results
		# self.sample_z = self.sampler.get_sample(self.batch_size, self.z_dim, 10)  # np.random.uniform(-1, 1,
		# size=(self.batch_size, self.z_dim))
		self.test_labels = self.data_y[0:self.batch_size]
		self.test_codes = np.concatenate((self.test_labels, np.zeros([self.batch_size, self.len_continuous_code])), axis=1)

		# saver to save model
		self.saver = tf.train.Saver()

		# summary writer
		self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)

		# restore check-point if it exits
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		if could_load:
			start_epoch = (int)(checkpoint_counter / self.num_batches)
			start_batch_id = checkpoint_counter - start_epoch * self.num_batches
			counter = checkpoint_counter
			print(" [*] Load SUCCESS")
		else:
			start_epoch = 0
			start_batch_id = 1
			counter = 1
			print(" [!] Load failed...")

		# loop for epoch
		start_time = time.time()
		for epoch in range(start_epoch, self.epoch):

			# get batch data
			for idx in range(start_batch_id, self.num_batches):
				batch_images = self.data_X[idx * self.batch_size:(idx + 1) * self.batch_size]
				self.test_batch_images = self.data_X[0 * self.batch_size:(0 + 1) * self.batch_size]

				# generate code
				if self.SUPERVISED == True:
					batch_labels = self.data_y[idx * self.batch_size:(idx + 1) * self.batch_size]
				else:
					# batch_labels = _multivariate_dist(self.batch_size, self.z_dim, 10)
					batch_labels = np.random.multinomial(1, self.len_discrete_code * [float(1.0 / self.len_discrete_code)],
					                                     size=[self.batch_size])

				batch_codes = np.concatenate((batch_labels, np.random.uniform(-1, 1, size=(self.batch_size, 2))), axis=1)
				# batch_codes = np.concatenate((batch_labels, _multivariate_dist(self.batch_size, 2, 2)), axis=1)
				batch_z_unif = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
				# batch_z = self.sampler.get_sample(self.batch_size, self.z_dim, 10)

				# update AE
				_, ae_loss, ae_summ, embedding = self.sess.run([self.ae_optim, self.ae_loss, self.ae_sum, self.embedding],
				                                               feed_dict={self.x: batch_images})
				self.writer.add_summary(ae_summ, counter)

				# update D network
				_, summary_str, d_loss = self.sess.run([self.d_optim, self.d_sum, self.d_loss],
				                                                                      feed_dict={self.x: batch_images, self.y: batch_codes, self.z: embedding})
				self.writer.add_summary(summary_str, counter)

				# update G and Q network
				_, summary_str_g, g_loss, _, summary_str_q, q_loss = self.sess.run(
					[self.g_optim, self.g_sum, self.g_loss, self.q_optim, self.q_sum, self.q_loss],
					feed_dict={self.x: batch_images, self.z: embedding, self.y: batch_codes})
				self.writer.add_summary(summary_str_g, counter)
				self.writer.add_summary(summary_str_q, counter)

				# display training status
				counter += 1
				print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, ae_loss: %.8f" % (
					epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss, ae_loss))

				# save training results for every 300 steps
				if np.mod(counter, 1) == 0:
					samples = self.sess.run(self.fake_images, feed_dict={self.z: embedding, self.y: self.test_codes})
					tot_num_samples = min(self.sample_num, self.batch_size)
					manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
					manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
					save_images(samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w], './' + check_folder(
						self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_train_{:02d}_{:04d}.png'.format(epoch, idx))

			# After an epoch, start_batch_id is set to zero
			# non-zero value is only for the first epoch after loading pre-trained model
			start_batch_id = 0

			# save model
			self.save(self.checkpoint_dir, counter)

			# show temporal results
			self.visualize_results(epoch)
		self.plot_train_test_loss("confidence", self.confidence_list)

		# save model for final step
		self.save(self.checkpoint_dir, counter)

	def visualize_results(self, epoch):
		tot_num_samples = min(self.sample_num, self.batch_size)
		image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

		""" random noise, random discrete code, fixed continuous code """
		y = np.random.choice(self.len_discrete_code, self.batch_size)
		y_one_hot = np.zeros((self.batch_size, self.y_dim))
		y_one_hot[np.arange(self.batch_size), y] = 1

		# z_sample = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
		z_sample = self.sampler.get_sample(self.batch_size, self.z_dim, 10)
		_, ae_loss, ae_summ, embedding_test = self.sess.run([self.ae_optim, self.ae_loss, self.ae_sum, self.embedding],
		                                                    feed_dict={self.x: self.test_batch_images})
		samples = self.sess.run(self.fake_images, feed_dict={self.z: embedding_test, self.y: y_one_hot})
		accuracy, confidence, loss = self.pretrained_classifier.test(samples.reshape(-1, self.input_width * self.input_height),
		                                                             np.ones((self.batch_size, self.len_discrete_code)), epoch)
		self.confidence_list.append(confidence)
		save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim], check_folder(
			self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + '_test_all_classes.png')

		""" specified condition, random noise """
		n_styles = 10  # must be less than or equal to self.batch_size

		np.random.seed()
		si = np.random.choice(self.batch_size, n_styles)

		for l in range(self.len_discrete_code):
			y = np.zeros(self.batch_size, dtype=np.int64) + l
			y_one_hot = np.zeros((self.batch_size, self.y_dim))
			y_one_hot[np.arange(self.batch_size), y] = 1

			samples = self.sess.run(self.fake_images, feed_dict={self.z: embedding_test, self.y: y_one_hot})
			# save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
			#             check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + '_test_class_%d.png' % l)

			samples = samples[si, :, :, :]

			if l == 0:
				all_samples = samples
			else:
				all_samples = np.concatenate((all_samples, samples), axis=0)

		""" save merged images to check style-consistency """
		canvas = np.zeros_like(all_samples)
		for s in range(n_styles):
			for c in range(self.len_discrete_code):
				canvas[s * self.len_discrete_code + c, :, :, :] = all_samples[c * n_styles + s, :, :, :]

		save_images(canvas, [n_styles, self.len_discrete_code], check_folder(
			self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + '_test_all_classes_style_by_style.png')

		""" fixed noise """
		assert self.len_continuous_code == 2

		c1 = np.linspace(-1, 1, image_frame_dim)
		c2 = np.linspace(-1, 1, image_frame_dim)
		xv, yv = np.meshgrid(c1, c2)
		xv = xv[:image_frame_dim, :image_frame_dim]
		yv = yv[:image_frame_dim, :image_frame_dim]

		c1 = xv.flatten()
		c2 = yv.flatten()

		z_fixed = np.zeros([self.batch_size, self.z_dim])

		for l in range(self.len_discrete_code):
			y = np.zeros(self.batch_size, dtype=np.int64) + l  # ones in the discrete_code idx * batch_size
			y_one_hot = np.zeros((self.batch_size, self.y_dim))
			y_one_hot[np.arange(self.batch_size), y] = 1
			# cartesian multiplication of the two latent codes
			y_one_hot[np.arange(image_frame_dim * image_frame_dim), self.len_discrete_code] = c1
			y_one_hot[np.arange(image_frame_dim * image_frame_dim), self.len_discrete_code + 1] = c2

			samples = self.sess.run(self.fake_images, feed_dict={self.z: z_fixed, self.y: y_one_hot})
			samples_2 = self.sess.run(self.fake_images, feed_dict={self.z: embedding_test, self.y: y_one_hot})

			save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim], check_folder(
				self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + '_test_class_c1c2_%d.png' % l)
			save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim], check_folder(
				self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch +
			            '_test_class_c1c2_%d_with_prior.png' % l)


	@property
	def model_dir(self):
		return "{}_{}_{}_{}".format(self.model_name, self.dataset_name, self.batch_size, self.z_dim)

	def save(self, checkpoint_dir, step):
		checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

	def load(self, checkpoint_dir):
		import re
		print(" [*] Reading checkpoints...")
		checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
			print(" [*] Success to read {}".format(ckpt_name))
			return True, counter
		else:
			print(" [*] Failed to find a checkpoint")
			return False, 0

	def plot_train_test_loss(self, name_of_measure, array, color="b",marker="P"):
		plt.Figure()
		plt.title('{} {} score'.format(self.dataset_name, name_of_measure), fontsize=18)
		x_range = np.linspace(1, len(array)-1, len(array))

		confidence, = plt.plot(x_range, array, color=color,marker=marker, label=name_of_measure, linewidth=2)
		plt.legend(handler_map={confidence: HandlerLine2D(numpoints=1)})
		plt.legend(bbox_to_anchor=(1.05, 1), loc=0, borderaxespad=0.)
		plt.yscale('linear')
		plt.xlabel('Epoch')
		plt.ylabel('Score')
		plt.grid()
		plt.show()
		plt.savefig("AEMultiModalInfoGAN_{}_{}_{}".format(self.dataset_name, type(self.sampler).__name__, name_of_measure))

		plt.close()