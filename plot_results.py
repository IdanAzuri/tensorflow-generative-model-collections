import sys

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import pickle

from matplotlib.legend_handler import HandlerLine2D


dir = 'classifier_results_seed_12/'
dir2 = 'classifier_results_seed_88/'
dir3 = 'classifier_results_seed_125/'
START = 3
start = START
END = 50


# 10 modalities
def fashion_MM_plot_from_pkl():
	import numpy as np
	import matplotlib.pyplot as plt
	import pickle
	# plt.Figure(figsize=(15, 15))
	
	a = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_GaussianSample_mu_0.0_sigma_0.2_czcc_czrc_rzcc_rzrc_accuracy.pkl", "rb"))[START:END])
	b = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	c = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_UniformSample_mu_0.0_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	d = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultiModalUniformSample_mu_0.0_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	
	a2 = np.mean(pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_GaussianSample_mu_0.0_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	b2 = np.mean(pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	c2 = np.mean(pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_UniformSample_mu_0.0_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	d2 = np.mean(pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultiModalUniformSample_mu_0.0_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	
	# DIR 3
	a3 = np.mean(pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_GaussianSample_mu_0.0_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	b3 = np.mean(pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	c3 = np.mean(pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_UniformSample_mu_0.0_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	d3 = np.mean(pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultiModalUniformSample_mu_0.0_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	
	# CALC MEAN AND STDERR
	a_mean = np.mean([a, a2, a3], axis=0)
	a_stderr = np.asarray(np.std([a, a2, a3], axis=0) / np.sqrt(3))
	b_mean = np.mean([b, b2, b3], axis=0)
	b_stderr = np.asarray(np.std([b, b2, b3], axis=0) / np.sqrt(3))
	c_mean = np.mean([c, c2, c3], axis=0)
	c_stderr = np.asarray(np.std([c, c2, c3], axis=0) / np.sqrt(3))
	d_mean = np.mean([d, d2, d3], axis=0)
	d_stderr = np.asarray(np.std([d, d2, d3], axis=0) / np.sqrt(3))
	
	fig, ax = plt.subplots()
	models = ['Gaussian 1d', 'Gaussian multi-modal', 'Uniform', 'Unifrom multi-modal ']
	plt.title('MMinfoGAN Fashion-Mnist Different Priros Accuracy (standard error)', fontsize=12)
	x_pos = np.arange(len(models))
	ax.bar(x_pos, [a_mean, b_mean, c_mean, d_mean], yerr=[a_stderr, b_stderr, c_stderr, d_stderr], align='center', alpha=0.5, ecolor='black', capsize=10)
	ax.set_xticks(x_pos)
	ax.set_xticklabels(models)
	ax.yaxis.grid(True)
	
	# Save the figure and show
	plt.tight_layout()
	plt.ylabel("Accuracy Score")
	plt.savefig("MMinfoGAN_mnist_MultivariateGaussianSampler.png")
	plt.show()
	plt.close()


def MM_mu_05_07_08_zoom_plot_from_pkl():
	import numpy as np
	import matplotlib.pyplot as plt
	import pickle
	plt.Figure(figsize=(15, 15))
	
	# title = 'MMinfoGAN_Fsion-Mnist_multi-modal Gaussian Sampler 10 modals'
	# plt.title(title, fontsize=10)
	a = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.5_sigma_0.17_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	b = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	c = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	d = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	e = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	f = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	g = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	# DIR 2
	a2 = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.5_sigma_0.17_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	b2 = np.mean(pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	c2 = np.mean(pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	d2 = np.mean(pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	e2 = np.mean(pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	f2 = np.mean(pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	g2 = np.mean(pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	
	# DIR 3
	# a3 = np.mean(pickle.load(
	# 	open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.5_sigma_0.17_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	# b3 = np.mean(pickle.load(
	# 	open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	c3 = np.mean(pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	d3 = np.mean(pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	# e3 = np.mean(pickle.load(
	# 	open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	f3 = np.mean(pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	g3 = np.mean(pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	
	# CALC MEAN AND STDERR
	a_mean = np.mean([a, a2], axis=0)
	a_stderr = np.std([a, a2], axis=0) / np.sqrt(3)
	b_mean = np.mean([b, b2], axis=0)
	b_stderr = np.std([b, b2], axis=0) / np.sqrt(3)
	c_mean = np.mean([c, c2, c3], axis=0)
	c_stderr = np.std([c, c2, c3], axis=0) / np.sqrt(3)
	d_mean = np.mean([d, d2, d3], axis=0)
	d_stderr = np.std([d, d2, d3], axis=0) / np.sqrt(3)
	e_mean = np.mean([e, e2], axis=0)
	e_stderr = np.std([e, e2], axis=0) / np.sqrt(3)
	f_mean = np.mean([f, f2, f3], axis=0)
	f_stderr = np.std([f, f2, f3], axis=0) / np.sqrt(3)
	g_mean = np.mean([g, g2, g3], axis=0)
	g_stderr = np.std([g, g2, g3], axis=0) / np.sqrt(3)
	
	fig, ax = plt.subplots()
	models = ["$\Sigma=0.15,\mu=0.7$", '$\Sigma=0.2,\mu=0.8$', '$\Sigma=0.3,\mu=0.8$']
	title = 'MMinfoGAN_Fsion-Mnist_multi-modal Gaussian Sampler 10 modals'
	ax.set_title(title, fontsize=10)
	x_pos = np.arange(len(models))
	ax.bar(x_pos, [c_mean, f_mean, g_mean], yerr=[c_stderr, f_stderr, g_stderr], align='center', alpha=0.5, ecolor='black', capsize=10)
	ax.set_ylabel('Accuracy')
	ax.set_xticks(x_pos)
	ax.set_xticklabels(models)
	# ax.set_title('Prior')
	ax.yaxis.grid(True)
	
	# Save the figure and show
	plt.tight_layout()
	
	plt.ylabel("Accuracy Score")
	plt.grid(True)
	plt.show()
	plt.savefig(title + ".png")
	plt.close()


def MM_mu_01_zoom_plot_from_pkl():
	import matplotlib.pyplot as plt
	import pickle
	
	a = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.1_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	b = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.13_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	c = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	d = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.17_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	# e = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	f = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	
	a2 = np.mean(pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.1_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	b2 = np.mean(pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.13_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	c2 = np.mean(pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	d2 = np.mean(pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.17_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	# e2 = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	f2 = np.mean(pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	
	a3 = np.mean(pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.1_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	# b3 = np.mean(pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.13_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	# c3 = np.mean(pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	d3 = np.mean(pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.17_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	# e3 = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	f3 = np.mean(pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	## CALC MEAN AND STDERR
	a_mean = np.mean([a, a2], axis=0)
	a_stderr = np.std([a, a2], axis=0) / np.sqrt(3)
	# b_mean = np.mean([b, b2], axis=0)
	# b_stderr = np.std([b, b2], axis=0) / np.sqrt(3)
	# c_mean = np.mean([c, c2, c3], axis=0)
	# c_stderr = np.std([c, c2, c3], axis=0) / np.sqrt(3)
	d_mean = np.mean([d, d2, d3], axis=0)
	d_stderr = np.std([d, d2, d3], axis=0) / np.sqrt(3)
	# e_mean = np.mean([e, e2], axis=0)
	# e_stderr = np.std([e, e2], axis=0) / np.sqrt(3)
	f_mean = np.mean([f, f2, f3], axis=0)
	f_stderr = np.std([f, f2, f3], axis=0) / np.sqrt(3)
	# g_mean = np.mean([g, g2, g3], axis=0)
	# g_stderr = np.std([g, g2, g3], axis=0) / np.sqrt(3)
	
	fig, ax = plt.subplots()
	models = ["$\Sigma=0.1,\mu=0.1$", '$\Sigma=0.17,\mu=0.1$', '$\Sigma=0.25,\mu=0.1$']
	title = 'MMinfoGAN_Fsion-Mnist_multi-modal Gaussian Sampler 10 modals mu=0.1'
	
	ax.set_title(title, fontsize=10)
	x_pos = np.arange(len(models))
	ax.bar(x_pos, [a_mean, d_mean, d_mean], yerr=[a_stderr, d_stderr, f_stderr], align='center', alpha=0.5, ecolor='black', capsize=10)
	ax.set_xticks(x_pos)
	ax.set_xticklabels(models)
	# ax.set_title('Prior')
	ax.yaxis.grid(True)
	
	# Save the figure and show
	plt.tight_layout()
	
	plt.ylabel("Accuracy Score")
	plt.grid(True)
	plt.show()
	plt.savefig(title + ".png")
	plt.close()


def MM_mu_1_zoom_plot_from_pkl():
	import matplotlib.pyplot as plt
	import pickle
	plt.Figure(figsize=(15, 15))
	
	# a = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.1_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	b = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	# c = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.22_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	# d = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	e = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	f = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.4_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	g = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.5_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	
	# a2 = np.mean(pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.1_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	b2 = np.mean(pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	# c = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.22_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	# d = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	e2 = np.mean(pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	f2 = np.mean(pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.4_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	g2 = np.mean(pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.5_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	#
	
	# a3 = np.mean(pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.1_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	b3 = np.mean(pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	# c = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.22_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	# d = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	e3 = np.mean(pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	f3 = np.mean(pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.4_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	g3 = np.mean(pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.5_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	## CALC MEAN AND STDERR
	# a_mean = np.mean([a, a2], axis=0)
	# a_stderr = np.std([a, a2], axis=0) / np.sqrt(3)
	b_mean = np.mean([b, b2,b3], axis=0)
	b_stderr = np.std([b, b2,b3], axis=0) / np.sqrt(3)
	# c_mean = np.mean([c, c2, c3], axis=0)
	# c_stderr = np.std([c, c2, c3], axis=0) / np.sqrt(3)
	# d_mean = np.mean([d, d2, d3], axis=0)
	# d_stderr = np.std([d, d2, d3], axis=0) / np.sqrt(3)
	e_mean = np.mean([e, e2,e3], axis=0)
	e_stderr = np.std([e, e2,e3], axis=0) / np.sqrt(3)
	# f_mean = np.mean([f, f2, f3], axis=0)
	# f_stderr = np.std([f, f2, f3], axis=0) / np.sqrt(3)
	g_mean = np.mean([g, g2, g3], axis=0)
	g_stderr = np.std([g, g2, g3], axis=0) / np.sqrt(3)
	
	fig, ax = plt.subplots()
	models = ["$\Sigma=0.2,\mu=1.0$", '$\Sigma=0.3,\mu=1.0$', '$\Sigma=0.5,\mu=1.0$']
	title = 'MMinfoGAN_Fsion-Mnist_multi-modal Gaussian Sampler 10 modals mu=1.0'
	
	ax.set_title(title, fontsize=10)
	x_pos = np.arange(len(models))
	ax.bar(x_pos, [b_mean, e_mean, g_mean], yerr=[b_stderr, e_stderr, g_stderr], align='center', alpha=0.5, ecolor='black', capsize=10)
	ax.set_xticks(x_pos)
	ax.set_xticklabels(models)
	# ax.set_title('Prior')
	ax.yaxis.grid(True)
	
	# Save the figure and show
	plt.tight_layout()
	
	plt.ylabel("Accuracy Score")
	plt.grid(True)
	plt.show()
	plt.savefig(title + ".png")
	plt.close()


# 5 modalities


def MM_mu_05_07_08_zoom_plot_from_pkl_5_modals():
	import numpy as np
	import matplotlib.pyplot as plt
	import pickle
	plt.Figure(figsize=(15, 15))
	
	# a = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.5_sigma_0.17_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	# b = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	c = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	d = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	# e = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	f = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	g = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	# DIR 2
	# a2 = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.5_sigma_0.17_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	# b2 = np.mean(pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	c2 = np.mean(pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	d2 = np.mean(pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	# e2 = np.mean(pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	f2 = np.mean(pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	g2 = np.mean(pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	
	# DIR 3
	# a3 = np.mean(pickle.load(
	# 	open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.5_sigma_0.17_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	# b3 = np.mean(pickle.load(
	# 	open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	c3 = np.mean(pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	d3 = np.mean(pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	# e3 = np.mean(pickle.load(
	# 	open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	f3 = np.mean(pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	g3 = np.mean(pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	## CALC MEAN AND STDERR
	# a_mean = np.mean([a, a2], axis=0)
	# a_stderr = np.std([a, a2], axis=0) / np.sqrt(3)
	# b_mean = np.mean([b, b2], axis=0)
	# b_stderr = np.std([b, b2], axis=0) / np.sqrt(3)
	c_mean = np.mean([c, c2, c3], axis=0)
	c_stderr = np.std([c, c2, c3], axis=0) / np.sqrt(3)
	# d_mean = np.mean([d, d2, d3], axis=0)
	# d_stderr = np.std([d, d2, d3], axis=0) / np.sqrt(3)
	# e_mean = np.mean([e, e2], axis=0)
	# e_stderr = np.std([e, e2], axis=0) / np.sqrt(3)
	f_mean = np.mean([f, f2, f3], axis=0)
	f_stderr = np.std([f, f2, f3], axis=0) / np.sqrt(3)
	g_mean = np.mean([g, g2, g3], axis=0)
	g_stderr = np.std([g, g2, g3], axis=0) / np.sqrt(3)
	
	fig, ax = plt.subplots()
	models = ["$\Sigma=0.15,\mu=0.7$", '$\Sigma=0.2,\mu=0.8$', '$\Sigma=0.3,\mu=0.8$']
	title = 'MMinfoGAN_Fsion-Mnist_multi-modal Gaussian Sampler 5 modals'
	
	ax.set_title(title, fontsize=10)
	x_pos = np.arange(len(models))
	ax.bar(x_pos, [c_mean, f_mean, g_mean], yerr=[c_stderr, f_stderr, g_stderr], align='center', alpha=0.5, ecolor='black', capsize=10)
	ax.set_ylabel('Accuracy')
	ax.set_xticks(x_pos)
	ax.set_xticklabels(models)
	# ax.set_title('Prior')
	ax.yaxis.grid(True)
	
	# Save the figure and show
	plt.tight_layout()
	
	plt.ylabel("Accuracy Score")
	plt.grid(True)
	plt.show()
	plt.savefig(title + ".png")
	plt.close()


def MM_mu_01_zoom_plot_from_pkl_5modals():
	import matplotlib.pyplot as plt
	import pickle
	
	a = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.1_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	# b = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.13_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	# c = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	d = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.17_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	# e = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	f = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	
	a2 = np.mean(pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.1_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	# b2 = np.mean(pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.13_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	# c2 = np.mean(pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	d2 = np.mean(pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.17_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	# e2 = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	f2 = np.mean(pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	
	a3 = np.mean(pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.1_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	# b3 = np.mean(pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.13_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	# c3 = np.mean(pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	d3 = np.mean(pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.17_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	# e3 = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	f3 = np.mean(pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	
	## CALC MEAN AND STDERR
	a_mean = np.mean([a, a2], axis=0)
	a_stderr = np.std([a, a2], axis=0) / np.sqrt(3)
	# b_mean = np.mean([b, b2], axis=0)
	# b_stderr = np.std([b, b2], axis=0) / np.sqrt(3)
	# c_mean = np.mean([c, c2, c3], axis=0)
	# c_stderr = np.std([c, c2, c3], axis=0) / np.sqrt(3)
	d_mean = np.mean([d, d2, d3], axis=0)
	d_stderr = np.std([d, d2, d3], axis=0) / np.sqrt(3)
	# e_mean = np.mean([e, e2], axis=0)
	# e_stderr = np.std([e, e2], axis=0) / np.sqrt(3)
	f_mean = np.mean([f, f2, f3], axis=0)
	f_stderr = np.std([f, f2, f3], axis=0) / np.sqrt(3)
	# g_mean = np.mean([g, g2, g3], axis=0)
	# g_stderr = np.std([g, g2, g3], axis=0) / np.sqrt(3)
	
	fig, ax = plt.subplots()
	models = ["$\Sigma=0.1,\mu=0.1$", '$\Sigma=0.17,\mu=0.1$', '$\Sigma=0.25,\mu=0.1$']
	title = 'MMinfoGAN_Fsion-Mnist_multi-modal Gaussian Sampler 5 modals mu=0.1'
	
	ax.set_title(title, fontsize=10)
	x_pos = np.arange(len(models))
	ax.bar(x_pos, [a_mean, d_mean, d_mean], yerr=[a_stderr, d_stderr, f_stderr], align='center', alpha=0.5, ecolor='black', capsize=10)
	ax.set_xticks(x_pos)
	ax.set_xticklabels(models)
	# ax.set_title('Prior')
	ax.yaxis.grid(True)
	
	# Save the figure and show
	plt.tight_layout()
	
	plt.ylabel("Accuracy Score")
	plt.grid(True)
	plt.show()
	plt.savefig(title + ".png")
	plt.close()



def MM_mu_1_zoom_plot_from_pkl_5modals():
	import matplotlib.pyplot as plt
	import pickle
	# a = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.1_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	b = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	# c = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.22_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	# d = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	e = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	f = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.4_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	g = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.5_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	
	# a2 = np.mean(pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.1_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	b2 = np.mean(pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	# c = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.22_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	# d = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	e2 = np.mean(pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	f2 = np.mean(pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.4_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	g2 = np.mean(pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.5_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	#
	
	# a3 = np.mean(pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.1_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	b3 = np.mean(pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	# c = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.22_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	# d = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	e3 = np.mean(pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	f3 = np.mean(pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.4_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	g3 = np.mean(pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.5_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END])
	#
	## CALC MEAN AND STDERR
	# a_mean = np.mean([a, a2, a3], axis=0)
	# a_stderr = np.std([a, a2, a3], axis=0) / np.sqrt(3)
	b_mean = np.mean([b, b2,b3], axis=0)
	b_stderr = np.std([b, b2,b3], axis=0) / np.sqrt(3)
	# c_mean = np.mean([c, c2, c3], axis=0)
	# c_stderr = np.std([c, c2, c3], axis=0) / np.sqrt(3)
	# d_mean = np.mean([d, d2, d3], axis=0)
	# d_stderr = np.std([d, d2, d3], axis=0) / np.sqrt(3)
	e_mean = np.mean([e, e2,e3], axis=0)
	e_stderr = np.std([e, e2, e3], axis=0) / np.sqrt(3)
	f_mean = np.mean([f, f2, f3], axis=0)
	f_stderr = np.std([f, f2, f3], axis=0) / np.sqrt(3)
	# g_mean = np.mean([g, g2, g3], axis=0)
	# g_stderr = np.std([g, g2, g3], axis=0) / np.sqrt(3)
	
	fig, ax = plt.subplots()
	models = ["$\Sigma=0.2,\mu=1.0$", '$\Sigma=0.3,\mu=1.0$', '$\Sigma=0.5,\mu=1.0$']
	title = 'MMinfoGAN_Fsion-Mnist_multi-modal Gaussian Sampler 5 modals mu=1.0'
	
	ax.set_title(title, fontsize=10)
	x_pos = np.arange(len(models))
	ax.bar(x_pos, [b_mean, e_mean, f_mean], yerr=[b_stderr, e_stderr, f_stderr], align='center', alpha=0.5, ecolor='black', capsize=10)
	ax.set_xticks(x_pos)
	ax.set_xticklabels(models)
	# ax.set_title('Prior')
	ax.yaxis.grid(True)
	
	# Save the figure and show
	plt.tight_layout()
	
	plt.ylabel("Accuracy Score")
	plt.grid(True)
	plt.show()
	plt.savefig(title + ".png")
	plt.close()



# 3 modalities


def MM_mu_05_07_08_zoom_plot_from_pkl3modals():
	import numpy as np
	import matplotlib.pyplot as plt
	import pickle
	
	# a = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.5_sigma_0.17_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	# b = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	c = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	# d = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	# e = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	f = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	g = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	# DIR 2
	# a2 = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.5_sigma_0.17_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	# b2 = np.mean(pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	c2 = np.mean(pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	d2 = np.mean(pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	# e2 = np.mean(pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	f2 = np.mean(pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	g2 = np.mean(pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	
	# DIR 3
	# a3 = np.mean(pickle.load(
	# 	open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.5_sigma_0.17_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	# b3 = np.mean(pickle.load(
	# 	open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	c3 = np.mean(pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	d3 = np.mean(pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	# e3 = np.mean(pickle.load(
	# 	open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	f3 = np.mean(pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	g3 = np.mean(pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	
	
	## CALC MEAN AND STDERR
	# a_mean = np.mean([a, a2], axis=0)
	# a_stderr = np.std([a, a2], axis=0) / np.sqrt(3)
	# b_mean = np.mean([b, b2], axis=0)
	# b_stderr = np.std([b, b2], axis=0) / np.sqrt(3)
	c_mean = np.mean([c, c2, c3], axis=0)
	c_stderr = np.std([c, c2, c3], axis=0) / np.sqrt(3)
	# d_mean = np.mean([d, d2, d3], axis=0)
	# d_stderr = np.std([d, d2, d3], axis=0) / np.sqrt(3)
	# e_mean = np.mean([e, e2], axis=0)
	# e_stderr = np.std([e, e2], axis=0) / np.sqrt(3)
	f_mean = np.mean([f, f2, f3], axis=0)
	f_stderr = np.std([f, f2, f3], axis=0) / np.sqrt(3)
	g_mean = np.mean([g, g2, g3], axis=0)
	g_stderr = np.std([g, g2, g3], axis=0) / np.sqrt(3)
	
	fig, ax = plt.subplots()
	models = ["$\Sigma=0.15,\mu=0.7$", '$\Sigma=0.2,\mu=0.8$', '$\Sigma=0.3,\mu=0.8$']
	title = 'MMinfoGAN_Fsion-Mnist_multi-modal Gaussian Sampler 3 modals'
	
	ax.set_title(title, fontsize=10)
	x_pos = np.arange(len(models))
	ax.bar(x_pos, [c_mean, f_mean, g_mean], yerr=[c_stderr, f_stderr, g_stderr], align='center', alpha=0.5, ecolor='black', capsize=10)
	ax.set_ylabel('Accuracy')
	ax.set_xticks(x_pos)
	ax.set_xticklabels(models)
	# ax.set_title('Prior')
	ax.yaxis.grid(True)
	
	# Save the figure and show
	plt.tight_layout()
	
	plt.ylabel("Accuracy Score")
	plt.grid(True)
	plt.show()
	plt.savefig(title + ".png")
	plt.close()


def MM_mu_01_zoom_plot_from_pkl3modals():
	import matplotlib.pyplot as plt
	import pickle
	
	a = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.1_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	# b = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.13_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	# c = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	d = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.17_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	# e = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	f = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	
	a2 = np.mean(pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.1_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	# b2 = np.mean(pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.13_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	# c2 = np.mean(pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	d2 = np.mean(pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.17_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	# e2 = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	f2 = np.mean(pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	
	a3 = np.mean(pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.1_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	# b3 = np.mean(pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.13_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	# c3 = np.mean(pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	d3 = np.mean(pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.17_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	# e3 = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	f3 = np.mean(pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	
	## CALC MEAN AND STDERR
	a_mean = np.mean([a, a2, a3], axis=0)
	a_stderr = np.std([a, a2, a3], axis=0) / np.sqrt(3)
	# b_mean = np.mean([b, b2], axis=0)
	# b_stderr = np.std([b, b2], axis=0) / np.sqrt(3)
	# c_mean = np.mean([c, c2, c3], axis=0)
	# c_stderr = np.std([c, c2, c3], axis=0) / np.sqrt(3)
	d_mean = np.mean([d, d2, d3], axis=0)
	d_stderr = np.std([d, d2, d3], axis=0) / np.sqrt(3)
	# e_mean = np.mean([e, e2], axis=0)
	# e_stderr = np.std([e, e2], axis=0) / np.sqrt(3)
	f_mean = np.mean([f, f2, f3], axis=0)
	f_stderr = np.std([f, f2, f3], axis=0) / np.sqrt(3)
	# g_mean = np.mean([g, g2, g3], axis=0)
	# g_stderr = np.std([g, g2, g3], axis=0) / np.sqrt(3)
	
	fig, ax = plt.subplots()
	models = ["$\Sigma=0.1,\mu=0.1$", '$\Sigma=0.17,\mu=0.1$', '$\Sigma=0.25,\mu=0.1$']
	title = 'MMinfoGAN_Fsion-Mnist_multi-modal Gaussian Sampler 3 modals mu=0.1'
	
	ax.set_title(title, fontsize=10)
	x_pos = np.arange(len(models))
	ax.bar(x_pos, [a_mean, d_mean, d_mean], yerr=[a_stderr, d_stderr, f_stderr], align='center', alpha=0.5, ecolor='black', capsize=10)
	ax.set_xticks(x_pos)
	ax.set_xticklabels(models)
	# ax.set_title('Prior')
	ax.yaxis.grid(True)
	
	# Save the figure and show
	plt.tight_layout()
	
	plt.ylabel("Accuracy Score")
	plt.grid(True)
	plt.show()
	plt.savefig(title + ".png")
	plt.close()


def MM_mu_1_zoom_plot_from_pkl3modals():
	import matplotlib.pyplot as plt
	import pickle
	# a = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.1_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END])
	b = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	# c = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.22_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	# d = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	e = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	# f = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.4_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	g = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.5_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	
	# a2 = np.mean(pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.1_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	b2 = np.mean(pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	# c = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.22_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	# d = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	e2 = np.mean(pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	f2 = np.mean(pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.4_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	g2 = np.mean(pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.5_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	#
	
	# a3 = np.mean(pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.1_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	b3 = np.mean(pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	# c = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.22_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	# d = np.mean(pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	e3 = np.mean(pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	f3 = np.mean(pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.4_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	g3 = np.mean(pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.5_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END])
	#
	## CALC MEAN AND STDERR
	# a_mean = np.mean([a, a2], axis=0)
	# a_stderr = np.std([a, a2], axis=0) / np.sqrt(3)
	b_mean = np.mean([b, b2,b3], axis=0)
	b_stderr = np.std([b, b2,b3], axis=0) / np.sqrt(3)
	# c_mean = np.mean([c, c2, c3], axis=0)
	# c_stderr = np.std([c, c2, c3], axis=0) / np.sqrt(3)
	# d_mean = np.mean([d, d2, d3], axis=0)
	# d_stderr = np.std([d, d2, d3], axis=0) / np.sqrt(3)
	e_mean = np.mean([e, e2,e3], axis=0)
	e_stderr = np.std([e, e2,e3], axis=0) / np.sqrt(3)
	# f_mean = np.mean([f, f2, f3], axis=0)
	# f_stderr = np.std([f, f2, f3], axis=0) / np.sqrt(3)
	g_mean = np.mean([g, g2, g3], axis=0)
	g_stderr = np.std([g, g2, g3], axis=0) / np.sqrt(3)
	
	fig, ax = plt.subplots()
	models = ["$\Sigma=0.2,\mu=1.0$", '$\Sigma=0.3,\mu=1.0$', '$\Sigma=0.5,\mu=1.0$']
	title = 'MMinfoGAN_Fsion-Mnist_multi-modal Gaussian Sampler 3 modals mu=1.0'
	
	ax.set_title(title, fontsize=10)
	x_pos = np.arange(len(models))
	ax.bar(x_pos, [b_mean, e_mean, g_mean], yerr=[b_stderr, e_stderr, g_stderr], align='center', alpha=0.5, ecolor='black', capsize=10)
	ax.set_xticks(x_pos)
	ax.set_xticklabels(models)
	# ax.set_title('Prior')
	ax.yaxis.grid(True)
	
	# Save the figure and show
	plt.tight_layout()
	
	plt.ylabel("Accuracy Score")
	plt.grid(True)
	plt.show()
	plt.savefig(title + ".png")
	plt.close()


if __name__ == '__main__':
	fashion_MM_plot_from_pkl()
	# plot_from_pkl()
	MM_mu_1_zoom_plot_from_pkl()
	MM_mu_01_zoom_plot_from_pkl()
	MM_mu_05_07_08_zoom_plot_from_pkl()
	MM_mu_05_07_08_zoom_plot_from_pkl_5_modals()
	MM_mu_1_zoom_plot_from_pkl_5modals()
	MM_mu_01_zoom_plot_from_pkl_5modals()
	MM_mu_1_zoom_plot_from_pkl3modals()
	MM_mu_01_zoom_plot_from_pkl3modals()
	MM_mu_05_07_08_zoom_plot_from_pkl3modals()
