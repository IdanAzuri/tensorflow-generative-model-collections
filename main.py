import argparse
import os

import matplotlib
matplotlib.use('Agg')
import tensorflow as tf

from ACGAN import ACGAN
## GAN Variants
from AEInfoGAN import AEMultiModalInfoGAN
from BEGAN import BEGAN
from CGAN import CGAN
from CVAE import CVAE
from DRAGAN import DRAGAN
from EBGAN import EBGAN
from GAN import GAN
from LSGAN import LSGAN
from MultiModalInfoGAN import MultiModalInfoGAN
from Sampler import *
## VAE Variants
from VAE import VAE
from WGAN import WGAN
from WGAN_GP import WGAN_GP
from infoGAN import infoGAN
from utils import check_folder
from utils import show_all_variables



"""parsing and configuration"""


def parse_args():
	desc = "Tensorflow implementation of GAN collections"
	parser = argparse.ArgumentParser(description=desc)

	parser.add_argument('--gan_type', type=str, default='GAN',
	                    choices=['GAN', 'CGAN', 'infoGAN', 'ACGAN', 'EBGAN', 'BEGAN', 'WGAN', 'WGAN_GP', 'DRAGAN', 'LSGAN', 'VAE', 'CVAE',
	                             'MultiModalInfoGAN', 'AEMultiModalInfoGAN'], help='The type of GAN', required=True)
	parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion-mnist', 'cifar10', 'celebA'], help='The name of '
	                                                                                                                          'dataset')
	parser.add_argument('--epoch', type=int, default=20, help='The number of epochs to run')
	parser.add_argument('--batch_size', type=int, default=32, help='The size of batch')
	parser.add_argument('--z_dim', type=int, default=62, help='Dimension of noise vector')
	parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', help='Directory name to save the checkpoints')
	parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the generated images')
	parser.add_argument('--log_dir', type=str, default='logs', help='Directory name to save training logs')
	parser.add_argument('--sampler', type=str, default='uniform',
	                    choices=['uniform', 'multi-uniform', 'multi-gaussian', 'multi-gaussianTF', 'gaussian','truncated'])
	parser.add_argument('--gpus', type=str, default='0')
	parser.add_argument('--len_continuous_code', type=int, default=2)
	parser.add_argument('--wgan', type=str, default=False)
	parser.add_argument('--mu', type=float, default=0.1)
	parser.add_argument('--sigma', type=float, default=0.15)

	return check_args(parser.parse_args())


"""checking arguments"""


def check_args(args):
	# --checkpoint_dir
	check_folder(args.checkpoint_dir)

	# --result_dir
	check_folder(args.result_dir)

	# --result_dir
	check_folder(args.log_dir)

	# --epoch
	assert args.epoch >= 1, 'number of epochs must be larger than or equal to one'

	# --batch_size
	assert args.batch_size >= 1, 'batch size must be larger than or equal to one'

	# --z_dim
	assert args.z_dim >= 1, 'dimension of noise vector must be larger than or equal to one'

	return args


"""main"""


def main():
	# parse arguments
	args = parse_args()
	if args is None:
		exit()

	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
	# open session
	models = [GAN, CGAN, infoGAN, ACGAN, EBGAN, WGAN, WGAN_GP, DRAGAN, LSGAN, BEGAN, VAE, CVAE, MultiModalInfoGAN, infoGAN,
	          AEMultiModalInfoGAN]
	len_continuous_code = args.len_continuous_code
	sampler = args.sampler
	mu=args.mu
	sigma=args.sigma
	sampler_method = UniformSample()
	if sampler == 'multi-uniform':
		sampler_method = MultiModalUniformSample()
	elif sampler == 'multi-gaussian':
		sampler= "{}/mu_{}_sigma{}".format(sampler,mu,sigma)
		sampler_method = MultivariateGaussianSampler()
	# elif sampler == 'multi-gaussianTF':
	# 	sampler= "{}/mu_{}_sigma{}".format(sampler,mu,sigma)
	# 	sampler_method = MultimodelGaussianTF(mu=mu,sigma=sigma)
	elif sampler == 'gaussian':
		sampler= "{}/mu_{}_sigma{}".format(sampler,mu,sigma)
		sampler_method = GaussianSample(mu=mu,sigma=sigma)
	elif sampler == 'truncated':
		sampler= "{}/mu_{}_sigma{}".format(sampler,mu,sigma)
		sampler_method = TruncatedGaussianSample(mu=mu,sigma=sigma)
	is_wgan_gp = args.wgan
	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

		gan = None
		for model in models:
			if args.gan_type == model.model_name:
				print("CHEKPOINT DIR: {}".format(sampler))
				gan = model(sess, epoch=args.epoch, batch_size=args.batch_size, z_dim=args.z_dim, dataset_name=args.dataset,
				            checkpoint_dir=args.checkpoint_dir + '/' + sampler, result_dir=args.result_dir + '/' + sampler,
				            log_dir=args.log_dir + '/' + sampler, sampler=sampler_method, is_wgan_gp=is_wgan_gp,len_continuous_code=len_continuous_code)
		if gan is None:
			raise Exception("[!] There is no option for " + args.gan_type)

		# build graph
		gan.build_model()

		# show network architecture
		show_all_variables()

		# launch the graph in a session
		gan.train()
		print(" [*] Training finished!")

		# visualize learned generator
		gan.visualize_results(args.epoch - 1)
		print(" [*] Testing finished!")


if __name__ == '__main__':
	main()
