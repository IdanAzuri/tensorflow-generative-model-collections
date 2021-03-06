# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import pickle
import time
import warnings

from sklearn.utils import shuffle


# SEED = 88

warnings.filterwarnings("ignore", category=RuntimeWarning)
from matplotlib.legend_handler import HandlerLine2D

import utils
from classifier import CNNClassifier, one_hot_encoder, CONFIDENCE_THRESHOLD
from ops import *
from utils import *
import tensorflow as tf

def gradient_penalty(real, fake, f):
	def interpolate(a, b):
		shape = tf.concat((tf.shape(a)[0:1], tf.tile([1], [a.shape.ndims - 1])), axis=0)
		alpha = tf.random_uniform(shape=shape, minval=0., maxval=1.)
		inter = a + alpha * (b - a)
		inter.set_shape(a.get_shape().as_list())
		return inter
	
	x = interpolate(real, fake)
	_, pred, _ = f(x)
	gradients = tf.gradients(pred, x)[0]
	# slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=range(1, x.shape.ndims)))
	# gp = tf.reduce_mean((slopes - 1.)**2)
	
	slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
	gp = tf.reduce_mean(tf.square(slopes - 1.))
	return gp


class MultiModalInfoGAN(object):
	model_name = "MultiModalInfoGAN"  # name for checkpoint
	
	def __init__(self, sess, epoch, batch_size, z_dim, dataset_name, checkpoint_dir, result_dir, log_dir, sampler, seed, len_continuous_code=2, is_wgan_gp=False,
	             dataset_creation_order=["czcc", "czrc", "rzcc", "rzrc"], SUPERVISED=True):
		print("saving to esults dir={}".format(result_dir))
		np.random.seed(seed)
		self.test_size = 5000
		self.wgan_gp = is_wgan_gp
		self.loss_list = []
		self.confidence_list = []
		self.sess = sess
		self.dataset_name = dataset_name
		self.checkpoint_dir = checkpoint_dir
		self.result_dir = result_dir
		self.log_dir = log_dir
		self.epoch = epoch
		self.batch_size = batch_size
		self.sampler = sampler
		self.pretrained_classifier = CNNClassifier(self.dataset_name, seed=seed)
		self.dataset_creation_order = dataset_creation_order
		self.SUPERVISED = SUPERVISED  # if it is true, label info is directly used for code
		self.dir_results = "classifier_results_seed_{}".format(seed)
		# train
		self.learning_rate = 0.0002
		self.beta1 = 0.5
		
		# test
		self.sample_num = 64  # number of generated images to be saved
		
		# code
		self.len_discrete_code = 10  # categorical distribution (i.e. label)
		self.len_continuous_code = len_continuous_code  # gaussian distribution (e.g. rotation, thickness)
		
		if dataset_name == 'mnist' or dataset_name == 'fashion-mnist':
			# parameters
			self.input_height = 28
			self.input_width = 28
			self.output_height = 28
			self.output_width = 28
			
			self.z_dim = z_dim  # dimension of noise-vector
			self.y_dim = self.len_discrete_code + self.len_continuous_code  # dimension of code-vector (label+two features)
			self.c_dim = 1
			
			# load mnist
			self.data_X, self.data_y = load_mnist(self.dataset_name)
			
			# get number of batches for a single epoch
			self.num_batches = len(self.data_X) // self.batch_size
		# elif dataset_name == 'cifar10':
		# 	print("IN CIFAR")
		# 	# parameters
		# 	self.input_height = 32
		# 	self.input_width = 32
		# 	self.output_height = 32
		# 	self.output_width = 32
		#
		# 	self.z_dim = z_dim  # dimension of noise-vector
		# 	self.y_dim = self.len_discrete_code + self.len_continuous_code  # dimension of code-vector (label+two features)
		# 	self.c_dim = 3
		# 	# self.data_X, self.data_y, self.test_x, self.test_labels = get_train_test_data()
		# 	# get number of batches for a single epoch
		# 	self.num_batches = len(self.data_X) // self.batch_size
		# elif dataset_name == 'celebA':
		# 	from data_load import preprocess_fn
		# 	print("in celeba")
		# 	img_paths = glob.glob('/Users/idan.a/data/celeba/*.jpg')
		# 	self.data_pool = utils.DiskImageData(img_paths, batch_size, shape=[218, 178, 3], preprocess_fn=preprocess_fn)
		# 	self.num_batches = len(self.data_pool) // (batch_size)
		#
		# 	# real_ipt = data_pool.batch()
		# 	# parameters
		# 	self.input_height = 64
		# 	self.input_width = 64
		# 	self.output_height = 32
		# 	self.output_width = 32
		#
		# 	self.z_dim = 128  # dimension of noise-vector
		# 	self.c_dim = 3
		# 	self.len_discrete_code = 100  # categorical distribution (i.e. label)
		# 	self.len_continuous_code = 0  # gaussian distribution (e.g. rotation, thickness)
		# 	self.y_dim = self.len_discrete_code + self.len_continuous_code  # dimension of code-vector (label+two features)
		# 	sess = utils.session()
		#
		# 	# iteration counter
		# 	it_cnt, update_cnt = utils.counter()
		#
		# 	sess.run(tf.global_variables_initializer())
		# 	sess.run(it_cnt)
		# 	sess.run(update_cnt)  # get number of batches for a single epoch
		self.model_dir = self.get_model_dir()
	
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
	
	def discriminator(self, x, is_training=True, reuse=True):
		# Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
		# Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
		if self.wgan_gp:
			with tf.variable_scope("wgan_discriminator", reuse=reuse):
				net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='d_conv1'))
				net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='d_conv2'), is_training=is_training, scope='d_bn2'))
				net = tf.reshape(net, [self.batch_size, -1])
				net = lrelu(bn(linear(net, 1024, scope='d_fc3'), is_training=is_training, scope='d_bn3'))
				out_logit = linear(net, 1, scope='d_fc4')
				out = tf.nn.sigmoid(out_logit)
		else:
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
			net = lrelu(bn(linear(net, 128 * self.input_height // 4 * self.input_width // 4, scope='g_fc2'), is_training=is_training, scope='g_bn2'))
			net = tf.reshape(net, [self.batch_size, int(self.input_height // 4), int(self.input_width // 4), 128])
			net = lrelu(
				bn(deconv2d(net, [self.batch_size, int(self.input_height // 2), int(self.input_width // 2), 64], 4, 4, 2, 2, name='g_dc3'), is_training=is_training, scope='g_bn3'))
			
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
		if self.wgan_gp:
			d_loss_real = - tf.reduce_mean(D_real_logits)
			d_loss_fake = tf.reduce_mean(D_fake_logits)
			
			self.d_loss = d_loss_real + d_loss_fake
			
			# get loss for generator
			self.g_loss = - d_loss_fake
			wd = tf.reduce_mean(D_real_logits) - tf.reduce_mean(D_fake_logits)
			gp = gradient_penalty(self.x, self.x_, self.discriminator)
			self.d_loss = -wd + gp * 10.0
			self.g_loss = -tf.reduce_mean(D_fake_logits)
		
		## 2. Information Loss
		code_fake, code_logit_fake = self.classifier(input4classifier_fake, is_training=True, reuse=False)
		# discrete code : categorical
		disc_code_est = code_logit_fake[:, :self.len_discrete_code]
		disc_code_tg = self.y[:, :self.len_discrete_code]
		q_disc_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=disc_code_est, labels=disc_code_tg))
		
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
			self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(self.d_loss, var_list=d_vars)
			self.g_optim = tf.train.AdamOptimizer(self.learning_rate * 5, beta1=self.beta1).minimize(self.g_loss, var_list=g_vars)
			self.q_optim = tf.train.AdamOptimizer(self.learning_rate * 5, beta1=self.beta1).minimize(self.q_loss, var_list=q_vars)
		
		"""" Testing """
		# for test
		self.fake_images = self.generator(self.z, self.y, is_training=False, reuse=True)
		""" Summary """
		d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
		d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
		d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
		g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
		
		q_loss_sum = tf.summary.scalar("g_loss", self.q_loss)
		q_disc_sum = tf.summary.scalar("q_disc_loss", q_disc_loss)
		q_cont_sum = tf.summary.scalar("q_cont_loss", q_cont_loss)
		
		# final summary operations
		self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
		self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])
		self.q_sum = tf.summary.merge([q_loss_sum, q_disc_sum, q_cont_sum])
	
	def train(self):
		
		# initialize all variables
		tf.global_variables_initializer().run()
		
		# graph inputs for visualize training results
		self.sample_z = self.sampler.get_sample(self.batch_size, self.z_dim, 10)  # np.random.uniform(-1, 1,
		# size=(self.batch_size, self.z_dim))
		self.test_labels = np.ones([self.batch_size, self.y_dim])
		if self.dataset_name != "celebA":
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
			start_batch_id = 0
			counter = 1
			print(" [!] Load failed...")
		
		# loop for epoch
		start_time = time.time()
		for epoch in range(start_epoch, self.epoch):
			# get batch data
			for idx in range(start_batch_id, self.num_batches):
				batch_images = self.data_X[idx * self.batch_size:(idx + 1) * self.batch_size]

				
				batch_labels = np.random.multinomial(1, self.len_discrete_code * [float(1.0 / self.len_discrete_code)], size=[self.batch_size])
				
				batch_codes = np.concatenate((batch_labels, np.random.uniform(-1, 1, size=(self.batch_size, self.len_continuous_code))), axis=1)
				batch_z = self.sampler.get_sample(self.batch_size, self.z_dim, 10)
				
				# update D network
				_, summary_str, d_loss = self.sess.run([self.d_optim, self.d_sum, self.d_loss], feed_dict={self.x: batch_images, self.y: batch_codes, self.z: batch_z})
				self.writer.add_summary(summary_str, counter)
				
				# update G and Q network
				_, summary_str_g, g_loss, _, summary_str_q, q_loss = self.sess.run([self.g_optim, self.g_sum, self.g_loss, self.q_optim, self.q_sum, self.q_loss],
				                                                                   feed_dict={self.x: batch_images, self.z: batch_z, self.y: batch_codes})
				self.writer.add_summary(summary_str_g, counter)
				self.writer.add_summary(summary_str_q, counter)
				
				# display training status
				counter += 1
				print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss,))
			
			# After an epoch, start_batch_id is set to zero
			# non-zero value is only for the first epoch after loading pre-trained model
			start_batch_id = 0
			
			# save model
			self.save(self.checkpoint_dir, counter)
			
			# show temporal results
			self.visualize_results(epoch)
		# plotting
		self.create_dataset_from_GAN()
		self.save(self.checkpoint_dir, counter)
	
	# if self.dataset_name != "celebA":
	# 	self.plot_train_test_loss("confidence", self.confidence_list)
	
	# Evaluation with classifier
	
	# save model for final step
	
	def visualize_results(self, epoch):
		tot_num_samples = min(self.sample_num, self.batch_size)
		image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))
		
		""" random noise, random discrete code, fixed continuous code """
		y = np.random.choice(self.len_discrete_code, self.batch_size)
		y_one_hot = np.zeros((self.batch_size, self.y_dim))
		y_one_hot[np.arange(self.batch_size), y] = 1
		z_sample = self.sampler.get_sample(self.batch_size, self.z_dim, 10)
		samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample, self.y: y_one_hot})  # samples_for_test.append(samples)
		
		save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
		            check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + '_test_all_classes.png')
		
		""" specified condition, random noise """
		n_styles = 10  # must be less than or equal to self.batch_size
		
		si = np.random.choice(self.batch_size, n_styles)
		
		for l in range(self.len_discrete_code):
			y = np.zeros(self.batch_size, dtype=np.int64) + l
			y_one_hot = np.zeros((self.batch_size, self.y_dim))
			y_one_hot[np.arange(self.batch_size), y] = 1
			
			samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample, self.y: y_one_hot})
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
		
		save_images(canvas, [n_styles, self.len_discrete_code],
		            check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + '_test_all_classes_style_by_style.png')
		
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
			
			save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
			            check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + '_test_class_c1c2_%d.png' % l)
	
	def create_dataset_from_GAN(self, is_confidence=False):
		
		generated_dataset = []
		generated_labels = []
		tot_num_samples = min(self.sample_num, self.batch_size)
		image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))
		c1 = np.linspace(-1, 1, image_frame_dim)
		c2 = np.linspace(-1, 1, image_frame_dim)
		xv, yv = np.meshgrid(c1, c2)
		xv = xv[:image_frame_dim, :image_frame_dim]
		yv = yv[:image_frame_dim, :image_frame_dim]
		
		c1 = xv.flatten()
		c2 = yv.flatten()
		datasetsize = self.test_size // self.batch_size
		generated_dataset_clean_z_clean_c = []
		generated_labels_clean_z_clean_c = []
		
		generated_dataset_clean_z_random_c = []
		generated_labels_clean_z_random_c = []
		
		generated_dataset_random_z_clean_c = []
		generated_labels_random_z_clean_c = []
		
		generated_dataset_random_z_random_c = []
		generated_labels_random_z_random_c = []
		for label in range(self.len_discrete_code):
			tmp = check_folder(self.result_dir + '/' + self.model_dir)
			
			for i in self.dataset_creation_order:
				num_iter = max(datasetsize // len(self.dataset_creation_order), 10)
				if i == 'czcc':
					for _ in range(num_iter):
						# clean samples z fixed - czcc
						z_fixed = np.zeros([self.batch_size, self.z_dim])
						y = np.zeros(self.batch_size, dtype=np.int64) + label  # ones in the discrete_code idx * batch_size
						y_one_hot = np.zeros((self.batch_size, self.y_dim))
						y_one_hot[np.arange(self.batch_size), y] = 1
						samples = self.sess.run(self.fake_images, feed_dict={self.z: z_fixed, self.y: y_one_hot})
						generated_dataset_clean_z_clean_c.append(samples.reshape(-1, 28, 28))  # storing generated images and label
						generated_labels_clean_z_clean_c += [label] * self.batch_size
						if _ == 1:
							save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
							            check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_type_czcc' + '_label_%d.png' % label)
				
				generated_dataset += generated_dataset_clean_z_clean_c
				generated_labels += generated_labels_clean_z_clean_c
				# print("adding czcc")
				if i == 'czrc':
					for _ in range(num_iter):
						# z fixed -czrc
						z_fixed = np.zeros([self.batch_size, self.z_dim])
						y = np.zeros(self.batch_size, dtype=np.int64) + label  # ones in the discrete_code idx * batch_size
						y_one_hot = np.zeros((self.batch_size, self.y_dim))
						y_one_hot[np.arange(self.batch_size), y] = 1
						y_one_hot[np.arange(image_frame_dim * image_frame_dim), self.len_discrete_code] = c1
						y_one_hot[np.arange(image_frame_dim * image_frame_dim), self.len_discrete_code + 1] = c2
						samples = self.sess.run(self.fake_images, feed_dict={self.z: z_fixed, self.y: y_one_hot})
						generated_dataset_clean_z_random_c.append(samples.reshape(-1, 28, 28))  # storing generated images and label
						generated_labels_clean_z_random_c += [label] * self.batch_size
						# if _ == 1:
						# 	save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
						# 	            check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_type_czrc' + '_label_%d.png' % label)
				
				generated_dataset += generated_dataset_clean_z_random_c
				generated_labels += generated_labels_clean_z_random_c
				# print("adding czrc")
				if i == 'rzcc':
					for _ in range(num_iter):
						# z random c-clean - rzcc
						z_sample = self.sampler.get_sample(self.batch_size, self.z_dim, 10)
						y = np.zeros(self.batch_size, dtype=np.int64) + label  # ones in the discrete_code idx * batch_size
						y_one_hot = np.zeros((self.batch_size, self.y_dim))
						y_one_hot[np.arange(self.batch_size), y] = 1
						samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample, self.y: y_one_hot})
						generated_dataset_random_z_clean_c.append(samples.reshape(-1, 28, 28))  # storing generated images and label
						generated_labels_random_z_clean_c += [label] * self.batch_size
						# if _ == 1:
						# 	save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
						# 	            check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_type_rzcc' + '_label_%d.png' % label)
					generated_dataset += generated_dataset_random_z_clean_c
					generated_labels += generated_labels_random_z_clean_c
				if i == 'rzrc':
					for _ in range(num_iter):
						# rzrc
						z_sample = self.sampler.get_sample(self.batch_size, self.z_dim, 10)
						y = np.zeros(self.batch_size, dtype=np.int64) + label  # ones in the discrete_code idx * batch_size
						
						y_one_hot = np.zeros((self.batch_size, self.y_dim))
						y_one_hot[np.arange(self.batch_size), y] = 1
						y_one_hot[np.arange(image_frame_dim * image_frame_dim), self.len_discrete_code] = c1
						y_one_hot[np.arange(image_frame_dim * image_frame_dim), self.len_discrete_code + 1] = c2
						samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample, self.y: y_one_hot})
						
						generated_dataset_random_z_random_c.append(samples.reshape(-1, 28, 28))  # storing generated images and label
						generated_labels_random_z_random_c += [label] * self.batch_size
						# if _ == 1:
						# 	save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
						# 	            check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_type_rzrc' + '_label_%d.png' % label)
					
					generated_dataset += generated_dataset_random_z_random_c
					generated_labels += generated_labels_random_z_random_c
		
		####### PREPROCESS ####
		# if len(generated_dataset_clean_z_clean_c) > 0:
		# 	clean_dataset = generated_dataset_clean_z_clean_c
		# 	clean_labels = generated_labels_clean_z_clean_c
		# else:
		clean_dataset = generated_dataset
		clean_labels = generated_labels
		data_X_clean_part = np.asarray([y for x in clean_dataset for y in x]).reshape(-1, 28, 28)
		data_y_clean_part = np.asarray(clean_labels, dtype=np.int32).flatten()
		
		data_y_all = np.asarray(generated_labels, dtype=np.int32).flatten()
		import copy
		
		data_y_updateable = copy.deepcopy(data_y_all)
		# pretraind = CNNClassifier(self.dataset_name, original_dataset_name=self.dataset_name)
		
		for current_label in range(10):
			small_mask = data_y_clean_part == current_label
			mask = data_y_all == current_label
			data_X_for_current_label = np.asarray(data_X_clean_part[np.where(small_mask == True)]).reshape(-1, 784)
			
			limit = min(len(data_X_for_current_label) // 10, 2 ** 14)
			dummy_labels = one_hot_encoder(np.random.randint(0, 10, size=(limit)))  # no meaning for the labels
			_, confidence, _, arg_max = self.pretrained_classifier.test(data_X_for_current_label[:limit].reshape(-1, 784), dummy_labels.reshape(-1, 10), is_arg_max=True)
			if is_confidence:
				print("confidence:{}".format(confidence))
				high_confidence_threshold_indices = confidence >= CONFIDENCE_THRESHOLD
				if len(high_confidence_threshold_indices[high_confidence_threshold_indices]) > 0:
					arg_max = arg_max[high_confidence_threshold_indices]
					print("The length of high confidence:")
					print(len(high_confidence_threshold_indices[high_confidence_threshold_indices]))
			print(str(len(arg_max)) + " were taken")
			new_label = np.bincount(arg_max).argmax()
			print("Assinging:{}".format(new_label))
			# plt.title("old_label=" + str(current_label) + "new_label=" + str(new_label))
			# plt.imshow(data_X_for_current_label[0].reshape(28, 28))
			# plt.show()
			data_y_updateable[mask] = new_label
			print(np.bincount(arg_max))
		data_y_all = one_hot_encoder(data_y_updateable)
		order_str = '_'.join(self.dataset_creation_order)
		if not os.path.exists(self.dir_results):
			os.makedirs(self.dir_results)
		if self.wgan_gp:
			params = "mu_{}_sigma_{}_ndist_{}_WGAN".format(self.sampler.mu, self.sampler.sigma, self.sampler.n_distributions)
		else:
			params = "mu_{}_sigma_{}_ndist_{}".format(self.sampler.mu, self.sampler.sigma, self.sampler.n_distributions)
		
		fname_trainingset_edited = "edited_training_set_{}_{}_{}".format(self.dataset_name, type(self.sampler).__name__, params)
		fname_labeles_edited = "edited_labels_{}_{}_{}".format(self.dataset_name, type(self.sampler).__name__, params)
		generated_dataset = np.asarray(generated_dataset).reshape(-1, 784)
		generated_dataset, data_y_all = shuffle(generated_dataset, data_y_all, random_state=0)
		pickle.dump(generated_dataset, open("{}/{}.pkl".format(self.dir_results, fname_trainingset_edited), 'wb'))
		output_path = open("{}/{}.pkl".format(self.dir_results, fname_labeles_edited), 'wb')
		pickle.dump(data_y_all, output_path)
		
		fname_trainingset = "generated_training_set_{}_{}_{}".format(self.dataset_name, type(self.sampler).__name__, params)
		print("\n\nSAMPLES SIZE={},LABELS={},SAVED TRAINING SET {}{}\n\n".format(len(generated_dataset), len(generated_labels), self.dir_results, fname_trainingset))
		fname_labeles = "generated_labels_{}_{}_{}".format(self.dataset_name, type(self.sampler).__name__, params)
		pickle.dump(np.asarray(generated_dataset), open(self.dir_results + "/{}.pkl".format(fname_trainingset), 'wb'))
		# np.asarray(generated_labels).reshape(np.asarray(generated_dataset).shape[:2])
		pickle.dump(np.asarray(generated_labels), open(self.dir_results + "/{}.pkl".format(fname_labeles), 'wb'))
		
		return
	
	def get_model_dir(self):
		if self.wgan_gp:
			return "wgan_{}_{}_batch{}".format(self.model_name, self.dataset_name, self.batch_size)
		else:
			return "{}_{}_batch{}".format(self.model_name, self.dataset_name, self.batch_size)
	
	def save(self, checkpoint_dir, step):
		checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
		
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		
		self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)
	
	def load(self, checkpoint_dir):
		import re
		print(" [*] Reading checkpoints...")
		checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
		
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
	
	def plot_train_test_loss(self, name_of_measure, array, color="b", marker="P"):
		plt.Figure()
		plt.title('{} {} score'.format(self.dataset_name, name_of_measure), fontsize=18)
		x_range = np.linspace(1, len(array) - 1, len(array))
		
		confidence, = plt.plot(x_range, array, color=color, marker=marker, label=name_of_measure, linewidth=2)
		plt.legend(handler_map={confidence: HandlerLine2D(numpoints=1)})
		plt.legend(bbox_to_anchor=(1.05, 1), loc=0, borderaxespad=0.)
		plt.yscale('linear')
		plt.xlabel('Epoch')
		plt.ylabel('Score')
		plt.grid()
		plt.show()
		if self.wgan_gp:
			name_figure = "WGAN_MMWinfoGAN_{}_{}_{}".format(self.dataset_name, type(self.sampler).__name__, name_of_measure)
		else:
			name_figure = "MMinfoGAN_{}_{}_{}".format(self.dataset_name, type(self.sampler).__name__, name_of_measure)
		plt.savefig(name_figure)
		plt.close()
		pickle.dump(array, open("{}.pkl".format(name_figure), 'wb'))


def plot_from_pkl():
	import numpy as np
	
	import pickle
	plt.Figure(figsize=(15, 15))
	dir = '/Users/idan.a/results_21_5/'
	plt.title('Wgan Confidence Score Different Sampling Method', fontsize=14)
	a = pickle.load(open(dir + "WGAN_GP_fashion-mnist_MultiModalUniformSample_confidence.pkl", "rb"))
	b = pickle.load(open(dir + "WGAN_GP_fashion-mnist_MultivariateGaussianSampler_confidence.pkl", "rb"))
	c = pickle.load(open(dir + "WGAN_GP_fashion-mnist_UniformSample_confidence.pkl", "rb"))
	d = pickle.load(open(dir + "WGAN_GP_fashion-mnist_GaussianSample_confidence.pkl", "rb"))
	# plt.plot(a, np.arange(len(a)), 'r--',  b,np.arange(len(b)), 'b--',  c,np.arange(len(c)),'g^',d,np.arange(len(d)),"y--")
	a_range = np.arange(len(a))
	b_range = np.arange(len(b))
	c_range = np.arange(len(c))
	d_range = np.arange(len(d))
	aa, = plt.plot(a_range, a, color='b', marker="P", label="Multimodal Uniform Sample", linewidth=1)
	bb, = plt.plot(b_range, b, color='g', marker='p', label="Multimodal Gaussian Sample", linewidth=1)
	cc, = plt.plot(c_range, c, color='r', marker='^', label="Uniform Sample", linewidth=1)
	dd, = plt.plot(d_range, d, color='y', marker="o", label="Gaussian Sample", linewidth=1)
	mean_line = plt.plot(c_range, np.ones_like(d_range) * 0.92, label='Benchmark', linestyle='--')
	
	# plt.legend(handler_map={aa: HandlerLine2D(numpoints=1)})
	plt.legend([aa, bb, cc, dd], ["Multimodal Uniform ", "Multimodal Gaussian", "Uniform", "Gaussian"],
	           handler_map={aa: HandlerLine2D(numpoints=1), bb: HandlerLine2D(numpoints=1), cc: HandlerLine2D(numpoints=1), dd: HandlerLine2D(numpoints=1)
		
	                        }, loc='lower right')
	# plt.legend(bbox_to_anchor=(1.05, 1), loc=0, borderaxespad=0.)
	plt.xlabel("Epoch")
	plt.ylabel("Confidence Score")
	# plt.axis("auto")
	plt.grid(True)
	plt.show()
	plt.savefig("all_plots_fashion_mnist_WGAN.png")
	plt.close()


if __name__ == '__main__':
	plot_from_pkl()
