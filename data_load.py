from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import scipy.misc

import glob
import scipy

import utils

import tensorflow as tf


""" param """
epoch = 50
batch_size = 64
lr = 0.0002
z_dim = 100
n_critic = 5
gpu_id = 3

''' data '''
# you should prepare your own data in ./data/img_align_celeba
# celeba original size is [218, 178, 3]


def preprocess_fn(img):
	crop_size = 108
	re_size = 64
	img = tf.image.crop_to_bounding_box(img, (218 - crop_size) // 2, (178 - crop_size) // 2, crop_size, crop_size)
	img = tf.to_float(tf.image.resize_images(img, [re_size, re_size], method=tf.image.ResizeMethod.BICUBIC)) / 127.5 - 1
	return img


sess = utils.session()

# iteration counter
it_cnt, update_cnt = utils.counter()

sess.run(tf.global_variables_initializer())
sess.run(it_cnt)
sess.run(update_cnt)

img_paths = glob.glob('/Users/idan.a/data/celeba/*.jpg')

data_pool = utils.DiskImageData(img_paths, batch_size, shape=[218, 178, 3], preprocess_fn=preprocess_fn)
batch_epoch = len(data_pool) // (batch_size * n_critic)
real_ipt = data_pool.batch()
sess.run(it_cnt)
it_epoch=1
# save_dir="tmp/"
scipy.misc.imsave('sss.png', utils.immerge(real_ipt, 10, 10))

