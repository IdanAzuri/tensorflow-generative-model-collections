"""
Most codes from https://github.com/carpedm20/DCGAN-tensorflow
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip

import matplotlib.pyplot as plt
import scipy.misc
import tensorflow.contrib.slim as slim


def extract_data(filename, num_data, head_size, data_size):
	with gzip.open(filename) as bytestream:
		bytestream.read(head_size)
		buf = bytestream.read(data_size * num_data)
		data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
	return data


def load_mnist(dataset_name):
	data_dir = os.path.join("../data", dataset_name)

	def extract_data(filename, num_data, head_size, data_size):
		with gzip.open(filename) as bytestream:
			bytestream.read(head_size)
			buf = bytestream.read(data_size * num_data)
			data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
		return data

	data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
	trX = data.reshape((60000, 28, 28, 1))

	data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
	trY = data.reshape((60000))

	data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
	teX = data.reshape((10000, 28, 28, 1))

	data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
	teY = data.reshape((10000))

	trY = np.asarray(trY)
	teY = np.asarray(teY)

	X = np.concatenate((trX, teX), axis=0)
	y = np.concatenate((trY, teY), axis=0).astype(np.int)

	seed = 547
	np.random.seed(seed)
	np.random.shuffle(X)
	np.random.seed(seed)
	np.random.shuffle(y)

	y_vec = np.zeros((len(y), 10), dtype=np.float)
	for i, label in enumerate(y):
		y_vec[i, y[i]] = 1.0

	return X / 255., y_vec


def load_celeba(dataset_name):
	data_dir = os.path.join("../data", dataset_name)

	data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
	trX = data.reshape((60000, 28, 28, 1))

	data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
	trY = data.reshape((60000))

	data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
	teX = data.reshape((10000, 28, 28, 1))

	data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
	teY = data.reshape((10000))

	trY = np.asarray(trY)
	teY = np.asarray(teY)

	X = np.concatenate((trX, teX), axis=0)
	y = np.concatenate((trY, teY), axis=0).astype(np.int)

	seed = 547
	np.random.seed(seed)
	np.random.shuffle(X)
	np.random.seed(seed)
	np.random.shuffle(y)

	y_vec = np.zeros((len(y), 10), dtype=np.float)
	for i, label in enumerate(y):
		y_vec[i, y[i]] = 1.0


def check_folder(log_dir):
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	return log_dir


def show_all_variables():
	model_vars = tf.trainable_variables()
	slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def get_image(image_path, input_height, input_width, resize_height=64, resize_width=64, crop=True, grayscale=False):
	image = imread(image_path, grayscale)
	return transform(image, input_height, input_width, resize_height, resize_width, crop)


def save_images(images, size, image_path):
	return imsave(inverse_transform(images), size, image_path)


def imread(path, grayscale=False):
	if (grayscale):
		return scipy.misc.imread(path, flatten=True).astype(np.float)
	else:
		return scipy.misc.imread(path).astype(np.float)


def merge_images(images, size):
	return inverse_transform(images)


def merge(images, size):
	h, w = images.shape[1], images.shape[2]
	if (images.shape[3] in (3, 4)):
		c = images.shape[3]
		img = np.zeros((h * size[0], w * size[1], c))
		for idx, image in enumerate(images):
			i = idx % size[1]
			j = idx // size[1]
			img[j * h:j * h + h, i * w:i * w + w, :] = image
		return img
	elif images.shape[3] == 1:
		img = np.zeros((h * size[0], w * size[1]))
		for idx, image in enumerate(images):
			i = idx % size[1]
			j = idx // size[1]
			img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
		return img
	else:
		raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')


def imsave(images, size, path):
	image = np.squeeze(merge(images, size))
	return scipy.misc.imsave(path, image)


def center_crop(x, crop_h, crop_w, resize_h=64, resize_w=64):
	if crop_w is None:
		crop_w = crop_h
	h, w = x.shape[:2]
	j = int(round((h - crop_h) / 2.))
	i = int(round((w - crop_w) / 2.))
	return scipy.misc.imresize(x[j:j + crop_h, i:i + crop_w], [resize_h, resize_w])


def transform(image, input_height, input_width, resize_height=64, resize_width=64, crop=True):
	if crop:
		cropped_image = center_crop(image, input_height, input_width, resize_height, resize_width)
	else:
		cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
	return np.array(cropped_image) / 127.5 - 1.


def inverse_transform(images):
	return (images + 1.) / 2.


""" Drawing Tools """


# borrowed from https://github.com/ykwon0407/variational_autoencoder/blob/master/variational_bayes.ipynb
def save_scattered_image(z, id, z_range_x, z_range_y, name='scattered_image.jpg'):
	N = 10
	plt.figure(figsize=(8, 6))
	plt.scatter(z[:, 0], z[:, 1], c=np.argmax(id, 1), marker='o', edgecolor='none', cmap=discrete_cmap(N, 'jet'))
	plt.colorbar(ticks=range(N))
	axes = plt.gca()
	axes.set_xlim([-z_range_x, z_range_x])
	axes.set_ylim([-z_range_y, z_range_y])
	plt.grid(True)
	plt.savefig(name)


# borrowed from https://gist.github.com/jakevdp/91077b0cae40f8f8244a
def discrete_cmap(N, base_cmap=None):
	"""Create an N-bin discrete colormap from the specified input map"""

	# Note that if base_cmap is a string or None, you can simply do
	#    return plt.cm.get_cmap(base_cmap, N)
	# The following works for string, None, or a colormap instance:

	base = plt.cm.get_cmap(base_cmap)
	color_list = base(np.linspace(0, 1, N))
	cmap_name = base.name + str(N)
	return base.from_list(cmap_name, color_list, N)


# LOADING DATASETS


import os
import re
import scipy
import numpy as np
import tensorflow as tf

from collections import OrderedDict


def mkdir(paths):
	if not isinstance(paths, (list, tuple)):
		paths = [paths]
	for path in paths:
		path_dir, _ = os.path.split(path)
		if not os.path.isdir(path_dir):
			os.makedirs(path_dir)


def session(graph=None, allow_soft_placement=True, log_device_placement=False, allow_growth=True):
	""" return a Session with simple config """

	config = tf.ConfigProto(allow_soft_placement=allow_soft_placement, log_device_placement=log_device_placement)
	config.gpu_options.allow_growth = allow_growth
	return tf.Session(graph=graph, config=config)


def tensors_filter(tensors, filters, combine_type='or'):
	assert isinstance(tensors, (list, tuple)), '`tensors` shoule be a list or tuple!'
	assert isinstance(filters, (str, list, tuple)), '`filters` should be a string or a list(tuple) of strings!'
	assert combine_type == 'or' or combine_type == 'and', "`combine_type` should be 'or' or 'and'!"

	if isinstance(filters, str):
		filters = [filters]

	f_tens = []
	for ten in tensors:
		if combine_type == 'or':
			for filt in filters:
				if filt in ten.name:
					f_tens.append(ten)
					break
		elif combine_type == 'and':
			all_pass = True
			for filt in filters:
				if filt not in ten.name:
					all_pass = False
					break
			if all_pass:
				f_tens.append(ten)
	return f_tens


def trainable_variables(filters=None, combine_type='or'):
	t_var = tf.trainable_variables()
	if filters is None:
		return t_var
	else:
		return tensors_filter(t_var, filters, combine_type)


def summary(tensor_collection, summary_type=['mean', 'stddev', 'max', 'min', 'sparsity', 'histogram']):
	"""
	usage:
	1. summary(tensor)
	2. summary([tensor_a, tensor_b])
	3. summary({tensor_a: 'a', tensor_b: 'b})
	"""

	def _summary(tensor, name, summary_type=['mean', 'stddev', 'max', 'min', 'sparsity', 'histogram']):
		""" Attach a lot of summaries to a Tensor. """

		if name is None:
			# Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
			# session. This helps the clarity of presentation on tensorboard.
			name = re.sub('%s_[0-9]*/' % 'tower', '', tensor.name)
			name = re.sub(':', '-', name)

		with tf.name_scope('summary_' + name):
			summaries = []
			if len(tensor._shape) == 0:
				summaries.append(tf.summary.scalar(name, tensor))
			else:
				if 'mean' in summary_type:
					mean = tf.reduce_mean(tensor)
					summaries.append(tf.summary.scalar(name + '/mean', mean))
				if 'stddev' in summary_type:
					mean = tf.reduce_mean(tensor)
					stddev = tf.sqrt(tf.reduce_mean(tf.square(tensor - mean)))
					summaries.append(tf.summary.scalar(name + '/stddev', stddev))
				if 'max' in summary_type:
					summaries.append(tf.summary.scalar(name + '/max', tf.reduce_max(tensor)))
				if 'min' in summary_type:
					summaries.append(tf.summary.scalar(name + '/min', tf.reduce_min(tensor)))
				if 'sparsity' in summary_type:
					summaries.append(tf.summary.scalar(name + '/sparsity', tf.nn.zero_fraction(tensor)))
				if 'histogram' in summary_type:
					summaries.append(tf.summary.histogram(name, tensor))
			return tf.summary.merge(summaries)

	if not isinstance(tensor_collection, (list, tuple, dict)):
		tensor_collection = [tensor_collection]
	with tf.name_scope('summaries'):
		summaries = []
		if isinstance(tensor_collection, (list, tuple)):
			for tensor in tensor_collection:
				summaries.append(_summary(tensor, None, summary_type))
		else:
			for tensor, name in tensor_collection.items():
				summaries.append(_summary(tensor, name, summary_type))
		return tf.summary.merge(summaries)


def counter(scope='counter'):
	with tf.variable_scope(scope):
		counter = tf.Variable(0, dtype=tf.int32, name='counter')
		update_cnt = tf.assign(counter, tf.add(counter, 1))
		return counter, update_cnt


def load_checkpoint(checkpoint_dir, session, var_list=None):
	print(' [*] Loading checkpoint...')
	ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
	if ckpt and ckpt.model_checkpoint_path:
		ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
		ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
	try:
		restorer = tf.train.Saver(var_list)
		restorer.restore(session, ckpt_path)
		print(' [*] Loading successful! Copy variables from % s' % ckpt_path)
		return True
	except:
		print(' [*] No suitable checkpoint!')
		return False


def memory_data_batch(memory_data_dict, batch_size, preprocess_fns={}, shuffle=True, num_threads=16, min_after_dequeue=5000,
                      allow_smaller_final_batch=False, scope=None):
	"""
	memory_data_dict:
		for example
		{'img': img_ndarray, 'point': point_ndarray} or
		{'img': img_tensor, 'point': point_tensor}
		the value of each item of `memory_data_dict` is in shape of (N, ...)
	preprocess_fns:
		for example
		{'img': img_preprocess_fn, 'point': point_preprocess_fn}
	"""

	with tf.name_scope(scope, 'memory_data_batch'):
		fields = []
		tensor_dict = OrderedDict()
		for k in memory_data_dict:
			fields.append(k)
			tensor_dict[k] = tf.convert_to_tensor(memory_data_dict[k])  # the same dtype of the input data
		data_num = tensor_dict[k].get_shape().as_list()[0]

		# slice to single example, and since it's memory data, the `capacity` is set as data_num
		data_values = tf.train.slice_input_producer(list(tensor_dict.values()), shuffle=shuffle, capacity=data_num)
		data_keys = list(tensor_dict.keys())
		data_dict = {}
		for k, v in zip(data_keys, data_values):
			if k in preprocess_fns:
				data_dict[k] = preprocess_fns[k](v)
			else:
				data_dict[k] = v

		# batch datas
		if shuffle:
			capacity = min_after_dequeue + (num_threads + 1) * batch_size
			data_batch = tf.train.shuffle_batch(data_dict, batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue,
			                                    num_threads=num_threads, allow_smaller_final_batch=allow_smaller_final_batch)
		else:
			data_batch = tf.train.batch(data_dict, batch_size=batch_size, allow_smaller_final_batch=allow_smaller_final_batch)

		return data_batch, data_num, fields


class MemoryData:

	def __init__(self, memory_data_dict, batch_size, preprocess_fns={}, shuffle=True, num_threads=16, min_after_dequeue=5000,
	             allow_smaller_final_batch=False, scope=None):
		"""
		memory_data_dict:
			for example
			{'img': img_ndarray, 'point': point_ndarray} or
			{'img': img_tensor, 'point': point_tensor}
			the value of each item of `memory_data_dict` is in shape of (N, ...)
		preprocess_fns:
			for example
			{'img': img_preprocess_fn, 'point': point_preprocess_fn}
		"""

		self.graph = tf.Graph()  # declare ops in a separated graph
		with self.graph.as_default():
			# @TODO
			# There are some strange errors if the gpu device is the
			# same with the main graph, but cpu device is ok. I don't know why...
			with tf.device('/cpu:0'):
				self._batch_ops, self._data_num, self._fields = memory_data_batch(memory_data_dict, batch_size, preprocess_fns, shuffle,
				                                                                  num_threads, min_after_dequeue, allow_smaller_final_batch,
				                                                                  scope)

		print(' [*] MemoryData: create session!')
		self.sess = session(graph=self.graph)
		self.coord = tf.train.Coordinator()
		self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

	def __len__(self):
		return self._data_num

	def batch(self, fields=None):
		batch_data = self.sess.run(self._batch_ops)
		if fields is None:
			fields = self._fields
		if isinstance(fields, (list, tuple)):
			return [batch_data[field] for field in fields]
		else:
			return batch_data[fields]

	def fields(self):
		return self._fields

	def __del__(self):
		print(' [*] MemoryData: stop threads and close session!')
		self.coord.request_stop()
		self.coord.join(self.threads)
		self.sess.close()


def disk_image_batch(image_paths, batch_size, shape, preprocess_fn=None, shuffle=True, num_threads=16, min_after_dequeue=100,
                     allow_smaller_final_batch=False, scope=None):
	"""
	This function is suitable for bmp, jpg, png and gif files
	image_paths: string list or 1-D tensor, each of which is an iamge path
	preprocess_fn: single image preprocessing function
	"""

	with tf.name_scope(scope, 'disk_image_batch'):
		data_num = len(image_paths)

		# dequeue a single image path and read the image bytes; enqueue the whole file list
		_, img = tf.WholeFileReader().read(tf.train.string_input_producer(image_paths, shuffle=shuffle, capacity=data_num))
		img = tf.image.decode_image(img)

		# preprocessing
		img.set_shape(shape)
		if preprocess_fn is not None:
			img = preprocess_fn(img)

		# batch datas
		if shuffle:
			capacity = min_after_dequeue + (num_threads + 1) * batch_size
			img_batch = tf.train.shuffle_batch([img], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue,
			                                   num_threads=num_threads, allow_smaller_final_batch=allow_smaller_final_batch)
		else:
			img_batch = tf.train.batch([img], batch_size=batch_size, allow_smaller_final_batch=allow_smaller_final_batch)

		return img_batch, data_num


class DiskImageData:

	def __init__(self, image_paths, batch_size, shape, preprocess_fn=None, shuffle=True, num_threads=16, min_after_dequeue=100,
	             allow_smaller_final_batch=False, scope=None):
		"""
		This function is suitable for bmp, jpg, png and gif files
		image_paths: string list or 1-D tensor, each of which is an iamge path
		preprocess_fn: single image preprocessing function
		"""

		self.graph = tf.Graph()  # declare ops in a separated graph
		with self.graph.as_default():
			# @TODO
			# There are some strange errors if the gpu device is the
			# same with the main graph, but cpu device is ok. I don't know why...
			with tf.device('/cpu:0'):
				self._batch_ops, self._data_num = disk_image_batch(image_paths, batch_size, shape, preprocess_fn, shuffle, num_threads,
				                                                   min_after_dequeue, allow_smaller_final_batch, scope)

		print(' [*] DiskImageData: create session!')
		self.sess = session(graph=self.graph)
		self.coord = tf.train.Coordinator()
		self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

	def __len__(self):
		return self._data_num

	def batch(self):
		return self.sess.run(self._batch_ops)

	def __del__(self):
		print(' [*] DiskImageData: stop threads and close session!')
		self.coord.request_stop()
		self.coord.join(self.threads)
		self.sess.close()


def to_range(images, min_value=0.0, max_value=1.0, dtype=None):
	"""
	transform images from [-1.0, 1.0] to [min_value, max_value] of dtype
	"""
	assert np.min(images) >= -1.0 - 1e-5 and np.max(images) <= 1.0 + 1e-5 and (
				images.dtype == np.float32 or images.dtype == np.float64), 'The input images should be float64(32) and in the range of [-1.0, 1.0]!'
	if dtype is None:
		dtype = images.dtype
	return ((images + 1.) / 2. * (max_value - min_value) + min_value).astype(dtype)


def imwrite(image, path):
	""" save an [-1.0, 1.0] image """

	if image.ndim == 3 and image.shape[2] == 1:  # for gray image
		image = np.array(image, copy=True)
		image.shape = image.shape[0:2]
	return scipy.misc.imsave(path, to_range(image, 0, 255, np.uint8))


def immerge(images, row, col):
	"""
	merge images into an image with (row * h) * (col * w)
	`images` is in shape of N * H * W(* C=1 or 3)
	"""

	h, w = images.shape[1], images.shape[2]
	if images.ndim == 4:
		img = np.zeros((h * row, w * col, images.shape[3]))
	elif images.ndim == 3:
		img = np.zeros((h * row, w * col))
	for idx, image in enumerate(images):
		i = idx % col
		j = idx // col
		img[j * h:j * h + h, i * w:i * w + w, ...] = image

	return img
