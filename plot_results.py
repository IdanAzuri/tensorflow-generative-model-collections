import sys

import matplotlib


# matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import pickle

from matplotlib.legend_handler import HandlerLine2D


dir = 'classifier_results_seed_12/'
dir2 = 'classifier_results_seed_88/'
dir3 = 'classifier_results_seed_125/'
START = 1
start = START
END = 50

#10 modalities
def fashion_MM_plot_from_pkl():
	import numpy as np
	import matplotlib.pyplot as plt
	import pickle
	plt.Figure(figsize=(15, 15))
	
	plt.title('MMinfoGAN Fashion-Mnist Different Priros Accuracy (standard error)', fontsize=12)
	
	a = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_GaussianSample_mu_0.0_sigma_0.2_czcc_czrc_rzcc_rzrc_accuracy.pkl", "rb"))[START:END]
	b = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	c = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_UniformSample_mu_0.0_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	d = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultiModalUniformSample_mu_0.0_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	
	a2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_GaussianSample_mu_0.0_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	b2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	c2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_UniformSample_mu_0.0_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	d2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultiModalUniformSample_mu_0.0_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	
	# DIR 3
	a3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_GaussianSample_mu_0.0_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	b3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	c3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_UniformSample_mu_0.0_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	d3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultiModalUniformSample_mu_0.0_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	
	a_range = np.arange(START, len(a) + START)
	b_range = np.arange(START, len(b) + START)
	c_range = np.arange(START, len(c) + START)
	d_range = np.arange(START, len(d) + START)
	# e_range = np.arange(START, len(e) + START)
	# f_range = np.arange(START, len(f) + START)
	# g_range = np.arange(START, len(g) + START)
	
	# CALC MEAN AND STDERR
	a_mean = np.mean([a, a2, a3], axis=0)
	a_stderr = np.std([a, a2, a3], axis=0) / np.sqrt(len(a))
	b_mean = np.mean([b, b2, b3], axis=0)
	b_stderr = np.std([b, b2, b3], axis=0) / np.sqrt(len(b))
	c_mean = np.mean([c, c2, c3], axis=0)
	c_stderr = np.std([c, c2, c3], axis=0) / np.sqrt(len(c))
	d_mean = np.mean([d, d2, d3], axis=0)
	d_stderr = np.std([d, d2, d3], axis=0) / np.sqrt(len(d))
	# e_mean = np.mean([e, e2, e3], axis=0)
	# e_stderr = np.std([e, e2, e3], axis=0) / np.sqrt(len(e))
	# f_mean = np.mean([f, f2, f3], axis=0)
	# f_stderr = np.std([f, f2, f3], axis=0) / np.sqrt(len(f))
	# g_mean = np.mean([g, g2, g3], axis=0)
	# g_stderr = np.std([g, g2, g3], axis=0) / np.sqrt(len(g))
	plt.errorbar(b_range, b_mean, yerr=b_stderr, color='r', marker='+', label="Gaussian multi-modal", ls='--', capsize=5, capthick=1, ecolor='k')
	plt.errorbar(a_range, a_mean, yerr=a_stderr, color='b', ls='--', label="Gaussian 1d", marker='.', capsize=5, capthick=1, ecolor='k')
	plt.errorbar(c_range, c_mean, yerr=c_stderr, color='g', marker='*', label="Uniform", ls='--', capsize=5, capthick=1, ecolor='k')
	plt.errorbar(d_range, d_mean, yerr=d_stderr, color='c', marker='d', label="Unifrom multi-modal ", ls='--', capsize=5, capthick=1, ecolor='k')
	# plt.errorbar(a_range, e_mean, yerr=e_stderr, color='red', ls='--', marker='o', capsize=5, capthick=1, ecolor='black')
	# plt.errorbar(f_range, f_mean, yerr=f_stderr, color='b', marker='+', label="$\Sigma=0.2,\mu=0.8$", ls='--', capsize=5, capthick=1, ecolor='k')
	# plt.errorbar(g_range, g_mean, yerr=g_stderr, color='r', marker='d', label="$\Sigma=0.3,\mu=08$", ls='--', capsize=5, capthick=1, ecolor='k')
	
	# ee, = plt.plot(e_range, e, color='k', marker="P", label="$\mathbb{N}(\sigma=0.2,\mu=0.1$)", linewidth=0.5)
	# # ff, = plt.plot(f_range, f, color='b', marker='.', label="$\mathbb{N}(\Sigma=0.2,\mu=0$)", linewidth=0.5)
	# gg, = plt.plot(g_range, g, color='r', marker='d', label="Uniform", linewidth=0.5)
	# hh, = plt.plot(h_range, h, color='c', marker=".", label="Multi-modal uniform", linewidth=0.5)
	# # jj, = plt.plot(j_range, j, color='m', marker=".", label="$\sigma=0.1,\mu=0$", linewidth=0.5)
	# ii, = plt.plot(i_range, i, color='y', marker="^", label="$\mathbb{N}(\Sigma=0.17,\mu=0$)", linewidth=0.5)
	# # mean_line = plt.plot(e_range, np.ones_like(e_range) * 0.92, label='Benchmark', linestyle='--')
	
	# plt.legend(handler_map={aa: HandlerLine2D(numpoints=1)})
	# plt.legend([ee, ff, gg], ["Gaussian", "Multimodal Gaussian", "Uniform"],
	#            handler_map={ee: HandlerLine2D(numpoints=1), ff: HandlerLine2D(numpoints=1), gg: HandlerLine2D(numpoints=1)
	# 	 }, loc='middle right')
	plt.yticks(np.arange(0.4, 0.7, step=0.05))
	# plt.xticks(np.arange(0, 800, step=50))
	plt.legend(loc='best')
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy Score")
	# plt.axis("off")
	plt.grid(True)
	plt.show()
	plt.savefig("MMinfoGAN_mnist_MultivariateGaussianSampler.png")
	plt.close()


def MM_mu_05_07_08_zoom_plot_from_pkl():
	import numpy as np
	import matplotlib.pyplot as plt
	import pickle
	plt.Figure(figsize=(15, 15))
	
	title = 'MMinfoGAN_Fsion-Mnist_multi-modal Gaussian Sampler 10 modals'
	plt.title(title, fontsize=10)
	a = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.5_sigma_0.17_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	b = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	c = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	d = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	e = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	f = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	g = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	# DIR 2
	a2 = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.5_sigma_0.17_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	b2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	c2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	d2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	e2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	f2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	g2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	
	# DIR 3
	# a3 = pickle.load(
	# 	open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.5_sigma_0.17_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	# b3 = pickle.load(
	# 	open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	c3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	d3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	# e3 = pickle.load(
	# 	open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	f3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	g3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	a_range = np.arange(START, len(a) + START)
	b_range = np.arange(START, len(b) + START)
	c_range = np.arange(START, len(c) + START)
	d_range = np.arange(START, len(d) + START)
	e_range = np.arange(START, len(e) + START)
	f_range = np.arange(START, len(f) + START)
	g_range = np.arange(START, len(g) + START)
	
	# CALC MEAN AND STDERR
	a_mean = np.mean([a, a2], axis=0)
	a_stderr = np.std([a, a2], axis=0) / np.sqrt(len(a))
	b_mean = np.mean([b, b2], axis=0)
	b_stderr = np.std([b, b2], axis=0) / np.sqrt(len(a))
	c_mean = np.mean([c, c2, c3], axis=0)
	c_stderr = np.std([c, c2, c3], axis=0) / np.sqrt(len(c))
	d_mean = np.mean([d, d2, d3], axis=0)
	d_stderr = np.std([d, d2, d3], axis=0) / np.sqrt(len(d))
	e_mean = np.mean([e, e2], axis=0)
	e_stderr = np.std([e, e2], axis=0) / np.sqrt(len(e))
	f_mean = np.mean([f, f2, f3], axis=0)
	f_stderr = np.std([f, f2, f3], axis=0) / np.sqrt(len(f))
	g_mean = np.mean([g, g2, g3], axis=0)
	g_stderr = np.std([g, g2, g3], axis=0) / np.sqrt(len(g))
	# plt.errorbar(a_range, a_mean, yerr=a_stderr, color='y', ls='--', marker='p', capsize=5, capthick=1, ecolor='black')
	# plt.errorbar(b_range, b_mean, yerr=b_stderr, color='red', ls='--', marker='.', capsize=5, capthick=1, ecolor='black')
	plt.errorbar(c_range, c_mean, yerr=c_stderr, color='g', marker='d', label="$\Sigma=0.15,\mu=0.7$", ls='--', capsize=5, capthick=1, ecolor='k')
	# plt.errorbar(d_range, d_mean, yerr=d_stderr, color='c', marker='.', label="$\Sigma=0.25,\mu=0.7$", ls='--', capsize=5, capthick=1, ecolor='k')
	# plt.errorbar(e_range, e_mean, yerr=e_stderr, color='b', ls='--', marker='.', capsize=5, capthick=1, ecolor='black')
	plt.errorbar(f_range, f_mean, yerr=f_stderr, color='c', marker='*', label="$\Sigma=0.2,\mu=0.8$", ls='--', capsize=5, capthick=1, ecolor='k')
	plt.errorbar(g_range, g_mean, yerr=g_stderr, color='r', marker='.', label="$\Sigma=0.3,\mu=0.8$", ls='--', capsize=5, capthick=1, ecolor='k')
	
	# aa, = plt.plot(a_range, a, color='b', marker="P", label="$\Sigma=0.17,\mu=0.5$", linewidth=0.5)
	# bb, = plt.plot(b_range, b, color='g', marker='d', label="$\Sigma=0.15,\mu=0.7$", linewidth=0.5)
	# cc, = plt.plot(c_range, c, color='r', marker='^', label="$\Sigma=0.25,\mu=0.7$", linewidth=0.5)
	# dd, = plt.plot(d_range, d, color='r', marker=".", label="$\Sigma=0.3,\mu=0.7$", linewidth=0.5)
	# ee, = plt.plot(e_range, e, color='k', marker="P", label="$\Sigma=0.15,\mu=0.8$", linewidth=0.5)
	# ff, = plt.plot(f_range, f, color='b', marker='+', label="$\Sigma=0.2,\mu=0.8$", linewidth=0.5)
	# gg, = plt.plot(g_range, g, color='r', marker='d', label="$\Sigma=0.3,\mu=08$", linewidth=0.5)
	# hh, = plt.plot(h_range, h, color='c', marker=".", label="$\sigma=0.8,\mu=0.8$", linewidth=0.5)
	# ii, = plt.plot(i_range, i, color='y', marker="^", label="$\sigma=0.1,\mu=0$", linewidth=0.5)
	# jj, = plt.plot(j_range, j, color='m', marker=".", label="$\sigma=0.15,\mu=0$", linewidth=0.5)
	# kk, = plt.plot(k_range, k, color='g', marker="*", label="$\sigma=1.0,\mu=0$", linewidth=0.5)
	
	# plt.legend(handler_map={aa: HandlerLine2D(numpoints=1)})
	# plt.legend([aa, bb, cc, dd], ["Multimodal Uniform ", "Multimodal Gaussian", "Uniform", "Gaussian"],
	#            handler_map={aa: HandlerLine2D(numpoints=1), bb: HandlerLine2D(numpoints=1), cc: HandlerLine2D(numpoints=1),
	#                         dd: HandlerLine2D(numpoints=1)
	
	# }, loc='middle right')
	# plt.yticks(np.arange(0.5, 0.7, step=0.05))
	plt.yticks(np.arange(0.45, 0.65, step=0.05))
	# plt.xticks(np.arange(0, 800, step=50))
	plt.legend(loc='best')
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy Score")
	# plt.axis("auto")
	plt.grid(True)
	plt.show()
	plt.savefig(title + ".png")
	plt.close()


def MM_mu_01_zoom_plot_from_pkl():
	import matplotlib.pyplot as plt
	import pickle
	plt.Figure(figsize=(15, 15))
	
	title = 'MMinfoGAN_Fsion-Mnist_multi-modal Gaussian Sampler 10 modals mu=0.1'
	plt.title(title, fontsize=10)
	a = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.1_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	b = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.13_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	c = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	d = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.17_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	# e = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	f = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	
	a2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.1_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	b2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.13_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	c2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	d2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.17_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	# e2 = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	f2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	
	a3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.1_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	# b3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.13_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	# c3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	d3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.17_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	# e3 = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	f3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	
	a_range = np.arange(START, len(a) + START)
	b_range = np.arange(START, len(b) + START)
	c_range = np.arange(START, len(c) + START)
	d_range = np.arange(START, len(d) + START)
	# e_range = np.arange(START, len(e) + START)
	f_range = np.arange(START, len(f) + START)
	# g_range = np.arange(START, len(g) + START)
	
	# CALC MEAN AND STDERR
	a_mean = np.mean([a, a2, a3], axis=0)
	a_stderr = np.std([a, a2, a3], axis=0) / np.sqrt(len(a))
	b_mean = np.mean([b, b2], axis=0)
	b_stderr = np.std([b, b2], axis=0) / np.sqrt(len(a))
	c_mean = np.mean([c, c2], axis=0)
	c_stderr = np.std([c, c2], axis=0) / np.sqrt(len(c))
	d_mean = np.mean([d, d2, d3], axis=0)
	d_stderr = np.std([d, d2, d3], axis=0) / np.sqrt(len(d))
	# e_mean = np.mean([e, e2], axis=0)
	# e_stderr = np.std([e, e2], axis=0) / np.sqrt(len(e))
	f_mean = np.mean([f, f2, f3], axis=0)
	f_stderr = np.std([f, f2, f3], axis=0) / np.sqrt(len(f))
	# g_mean = np.mean([g, g2, g3], axis=0)
	# g_stderr = np.std([g, g2, g3], axis=0) / np.sqrt(len(g))
	plt.errorbar(a_range, a_mean, yerr=a_stderr, color='r', ls='--', label="$\Sigma=0.1,\mu=0.1$", marker='+', capsize=5, capthick=1, ecolor='black')
	# plt.errorbar(b_range, b_mean, yerr=b_stderr, color='r', ls='--', label="$\Sigma=0.13,\mu=0.1$",marker='.', capsize=5, capthick=1, ecolor='black')
	# plt.errorbar(c_range, c_mean, yerr=c_stderr, color='g', marker='.', label="$\Sigma=0.15,\mu=0.1$", ls='--', capsize=5, capthick=1, ecolor='k')
	plt.errorbar(d_range, d_mean, yerr=d_stderr, color='g', marker='+', label="$\Sigma=0.17,\mu=0.1$", ls='--', capsize=5, capthick=1, ecolor='k')
	# plt.errorbar(e_range, e_mean, yerr=e_stderr, color='b', ls='--', marker='.', capsize=5, capthick=1, ecolor='black')
	plt.errorbar(f_range, f_mean, yerr=f_stderr, color='c', marker='.', label="$\Sigma=0.25,\mu=0.1$", ls='--', capsize=5, capthick=1, ecolor='k')
	# plt.errorbar(g_range, g_mean, yerr=g_stderr, color='r', marker='.', label="$\Sigma=0.3,\mu=0.8$", ls='--', capsize=5, capthick=1, ecolor='k')
	
	#
	# aa, = plt.plot(a_range, a, color='b', marker="P", label="$\Sigma=0.1,\mu=0.1$", linewidth=0.5)
	# bb, = plt.plot(b_range, b, color='g', marker='d', label="$\Sigma=0.13,\mu=0.1$", linewidth=0.5)
	# cc, = plt.plot(c_range, c, color='r', marker='^', label="$\Sigma=0.15,\mu=0.1$", linewidth=0.5)
	# dd, = plt.plot(d_range, d, color='y', marker="+", label="$\Sigma=0.17,\mu=0.1$", linewidth=0.5)
	# ee, = plt.plot(e_range, e, color='k', marker="P", label="$\Sigma=0.2,\mu=0.1$", linewidth=0.5)
	# ff, = plt.plot(f_range, f, color='c', marker='.', label="$\Sigma=0.25,\mu=0.1$", linewidth=0.5)
	# gg, = plt.plot(g_range, g, color='r', marker='d', label="$\sigma=0.3,\mu=0$", linewidth=0.5)
	# hh, = plt.plot(h_range, h, color='c', marker=".", label="$\sigma=0.2,\mu=0$", linewidth=0.5)
	# ii, = plt.plot(i_range, i, color='y', marker="^", label="$\sigma=0.1,\mu=0$", linewidth=0.5)
	# jj, = plt.plot(j_range, j, color='m', marker=".", label="$\sigma=0.15,\mu=0$", linewidth=0.5)
	# kk, = plt.plot(k_range, k, color='g', marker="*", label="$\sigma=1.0,\mu=0$", linewidth=0.5)
	
	# plt.legend(handler_map={aa: HandlerLine2D(numpoints=1)})
	# plt.legend([aa, bb, cc, dd], ["Multimodal Uniform ", "Multimodal Gaussian", "Uniform", "Gaussian"],
	#            handler_map={aa: HandlerLine2D(numpoints=1), bb: HandlerLine2D(numpoints=1), cc: HandlerLine2D(numpoints=1),
	#                         dd: HandlerLine2D(numpoints=1)
	
	# }, loc='middle right')
	plt.yticks(np.arange(0.45, 0.65, step=0.05))
	plt.legend(loc='best')
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy Score")
	# plt.axis("auto")
	plt.grid(True)
	plt.show()
	plt.savefig(title + ".png")
	plt.close()


def MM_mu_1_zoom_plot_from_pkl():
	import matplotlib.pyplot as plt
	import pickle
	plt.Figure(figsize=(15, 15))
	
	title = 'MMinfoGAN_Fsion-Mnist_multi-modal Gaussian Sampler 10 modals mu=1.0'
	plt.title(title, fontsize=10)
	# a = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.1_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	b = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	# c = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.22_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	# d = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	e = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	f = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.4_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	g = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.5_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	
	# a2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.1_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	b2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	# c = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.22_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	# d = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	e2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	f2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.4_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	g2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.5_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	#
	
	# a3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.1_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	b3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	# c = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.22_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	# d = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	e3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	f3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.4_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	g3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.5_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	#
	
	# a_range = np.arange(START, len(a) + START)
	b_range = np.arange(START, len(b) + START)
	# c_range = np.arange(START, len(c) + START)
	# d_range = np.arange(START, len(d) + START)
	e_range = np.arange(START, len(e) + START)
	f_range = np.arange(START, len(f) + START)
	g_range = np.arange(START, len(g) + START)
	# CALC MEAN AND STDERR
	# a_mean = np.mean([a, a2], axis=0)
	# a_stderr = np.std([a, a2], axis=0) / np.sqrt(len(a))
	b_mean = np.mean([b, b2, b3], axis=0)
	b_stderr = np.std([b, b2, b3], axis=0) / np.sqrt(len(b))
	# c_mean = np.mean([c, c2, c3], axis=0)
	# c_stderr = np.std([c, c2, c3], axis=0) / np.sqrt(len(c))
	# d_mean = np.mean([d, d2, d3], axis=0)
	# d_stderr = np.std([d, d2, d3], axis=0) / np.sqrt(len(d))
	e_mean = np.mean([e, e2], axis=0)
	e_stderr = np.std([e, e2], axis=0) / np.sqrt(len(e))
	f_mean = np.mean([f, f2, f3], axis=0)
	f_stderr = np.std([f, f2, f3], axis=0) / np.sqrt(len(f))
	g_mean = np.mean([g, g2, g3], axis=0)
	g_stderr = np.std([g, g2, g3], axis=0) / np.sqrt(len(g))
	# plt.errorbar(a_range, a_mean, yerr=a_stderr, color='y', ls='--', marker='.', capsize=5, capthick=1, ecolor='black')
	plt.errorbar(b_range, b_mean, yerr=b_stderr, color='g', ls='--', label="$\Sigma=0.2,\mu=1.0$", marker='+', capsize=5, capthick=1, ecolor='black')
	# plt.errorbar(c_range, c_mean, yerr=c_stderr, color='g', marker='.', label="$\Sigma=0.15,\mu=0.7$", ls='--', capsize=5, capthick=1, ecolor='k')
	# plt.errorbar(d_range, d_mean, yerr=d_stderr, color='c', marker='.', label="$\Sigma=0.25,\mu=0.7$", ls='--', capsize=5, capthick=1, ecolor='k')
	plt.errorbar(e_range, e_mean, yerr=e_stderr, color='r', ls='--', label="$\Sigma=0.3,\mu=1.0$", marker='d', capsize=5, capthick=1, ecolor='black')
	# plt.errorbar(f_range, f_mean, yerr=f_stderr, color='g', marker='.', label="$\Sigma=0.4,\mu=1.0$", ls='--', capsize=5, capthick=1, ecolor='k')
	plt.errorbar(g_range, g_mean, yerr=g_stderr, color='c', marker='.', label="$\Sigma=0.5,\mu=1.0$", ls='--', capsize=5, capthick=1, ecolor='k')
	
	# aa, = plt.plot(a_range, a, color='b', marker="P", label="$\Sigma=0.15,\mu=1.0$", linewidth=0.5)
	# bb, = plt.plot(b_range, b, color='g', marker='d', label="$\Sigma=0.2,\mu=1.0$", linewidth=0.5)
	# # cc, = plt.plot(c_range, c, color='r', marker='^', label="$\sigma=0.22,\mu=1.0$", linewidth=0.5)
	# # dd, = plt.plot(d_range, d, color='y', marker=".", label="$\sigma=0.25,\mu=1.0$", linewidth=0.5)
	# ee, = plt.plot(e_range, e, color='k', marker="P", label="$\Sigma=0.3,\mu=1.0$", linewidth=0.5)
	# ff, = plt.plot(f_range, f, color='y', marker='^', label="$\Sigma=0.4,\mu=1.0$", linewidth=0.5)
	# gg, = plt.plot(g_range, g, color='r', marker='d', label="$\Sigma=0.5,\mu=1.0$", linewidth=0.5)
	# hh, = plt.plot(h_range, h, color='c', marker=".", label="$\sigma=0.2,\mu=1.0$", linewidth=0.5)
	# ii, = plt.plot(i_range, i, color='y', marker="^", label="$\sigma=0.1,\mu=0$", linewidth=0.5)
	# jj, = plt.plot(j_range, j, color='m', marker=".", label="$\sigma=0.15,\mu=0$", linewidth=0.5)
	# kk, = plt.plot(k_range, k, color='g', marker="*", label="$\sigma=1.0,\mu=0$", linewidth=0.5)
	
	# plt.legend(handler_map={aa: HandlerLine2D(numpoints=1)})
	# plt.legend([aa, bb, cc, dd], ["Multimodal Uniform ", "Multimodal Gaussian", "Uniform", "Gaussian"],
	#            handler_map={aa: HandlerLine2D(numpoints=1), bb: HandlerLine2D(numpoints=1), cc: HandlerLine2D(numpoints=1),
	#                         dd: HandlerLine2D(numpoints=1)
	
	# }, loc='middle right')
	plt.yticks(np.arange(0.45, 0.65, step=0.05))
	plt.legend(loc='best')
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy Score")
	# plt.axis("auto")
	plt.grid(True)
	plt.show()
	plt.savefig(title + ".png")
	plt.close()
#5 modalities


def MM_mu_05_07_08_zoom_plot_from_pkl_5_modals():
	import numpy as np
	import matplotlib.pyplot as plt
	import pickle
	plt.Figure(figsize=(15, 15))
	
	title = 'MMinfoGAN_Fsion-Mnist_multi-modal Gaussian Sampler 5 modals'
	plt.title(title, fontsize=10)
	# a = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.5_sigma_0.17_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	# b = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	c = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	d = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	# e = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	f = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	g = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	# DIR 2
	# a2 = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.5_sigma_0.17_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	# b2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	c2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	d2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	# e2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	f2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	g2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	
	# DIR 3
	# a3 = pickle.load(
	# 	open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.5_sigma_0.17_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	# b3 = pickle.load(
	# 	open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	c3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	d3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	# e3 = pickle.load(
	# 	open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	f3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	g3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	# a_range = np.arange(START, len(a) + START)
	# b_range = np.arange(START, len(b2) + START)
	c_range = np.arange(START, len(c) + START)
	d_range = np.arange(START, len(d) + START)
	# e_range = np.arange(START, len(e2) + START)
	f_range = np.arange(START, len(f) + START)
	g_range = np.arange(START, len(g) + START)
	
	# CALC MEAN AND STDERR
	# a_mean = np.mean([a, a2], axis=0)
	# a_stderr = np.std([a, a2], axis=0) / np.sqrt(len(a))
	# b_mean = np.mean([b2, b3], axis=0)
	# b_stderr = np.std([b2, b3], axis=0) / np.sqrt(len(a))
	c_mean = np.mean([c, c2, c3], axis=0)
	c_stderr = np.std([c, c2, c3], axis=0) / np.sqrt(len(c))
	d_mean = np.mean([d, d2, d3], axis=0)
	d_stderr = np.std([d, d2, d3], axis=0) / np.sqrt(len(d))
	# e_mean = np.mean([e3, e2], axis=0)
	# e_stderr = np.std([e3, e2], axis=0) / np.sqrt(len(e))
	f_mean = np.mean([f, f2, f3], axis=0)
	f_stderr = np.std([f, f2, f3], axis=0) / np.sqrt(len(f))
	g_mean = np.mean([g, g2, g3], axis=0)
	g_stderr = np.std([g, g2, g3], axis=0) / np.sqrt(len(g))
	# plt.errorbar(a_range, a_mean, yerr=a_stderr, color='y', ls='--', marker='p', capsize=5, capthick=1, ecolor='black')
	# plt.errorbar(b_range, b_mean, yerr=b_stderr, color='red', ls='--', marker='.', capsize=5, capthick=1, ecolor='black')
	plt.errorbar(c_range, c_mean, yerr=c_stderr, color='g', marker='d', label="$\Sigma=0.15,\mu=0.7$", ls='--', capsize=5, capthick=1, ecolor='k')
	# plt.errorbar(d_range, d_mean, yerr=d_stderr, color='c', marker='.', label="$\Sigma=0.25,\mu=0.7$", ls='--', capsize=5, capthick=1, ecolor='k')
	# plt.errorbar(e_range, e_mean, yerr=e_stderr, color='b', ls='--', marker='.', capsize=5, capthick=1, ecolor='black')
	plt.errorbar(f_range, f_mean, yerr=f_stderr, color='c', marker='*', label="$\Sigma=0.2,\mu=0.8$", ls='--', capsize=5, capthick=1, ecolor='k')
	plt.errorbar(g_range, g_mean, yerr=g_stderr, color='r', marker='.', label="$\Sigma=0.3,\mu=0.8$", ls='--', capsize=5, capthick=1, ecolor='k')
	
	# aa, = plt.plot(a_range, a, color='b', marker="P", label="$\Sigma=0.17,\mu=0.5$", linewidth=0.5)
	# bb, = plt.plot(b_range, b, color='g', marker='d', label="$\Sigma=0.15,\mu=0.7$", linewidth=0.5)
	# cc, = plt.plot(c_range, c, color='r', marker='^', label="$\Sigma=0.25,\mu=0.7$", linewidth=0.5)
	# dd, = plt.plot(d_range, d, color='r', marker=".", label="$\Sigma=0.3,\mu=0.7$", linewidth=0.5)
	# ee, = plt.plot(e_range, e, color='k', marker="P", label="$\Sigma=0.15,\mu=0.8$", linewidth=0.5)
	# ff, = plt.plot(f_range, f, color='b', marker='+', label="$\Sigma=0.2,\mu=0.8$", linewidth=0.5)
	# gg, = plt.plot(g_range, g, color='r', marker='d', label="$\Sigma=0.3,\mu=08$", linewidth=0.5)
	# hh, = plt.plot(h_range, h, color='c', marker=".", label="$\sigma=0.8,\mu=0.8$", linewidth=0.5)
	# ii, = plt.plot(i_range, i, color='y', marker="^", label="$\sigma=0.1,\mu=0$", linewidth=0.5)
	# jj, = plt.plot(j_range, j, color='m', marker=".", label="$\sigma=0.15,\mu=0$", linewidth=0.5)
	# kk, = plt.plot(k_range, k, color='g', marker="*", label="$\sigma=1.0,\mu=0$", linewidth=0.5)
	
	# plt.legend(handler_map={aa: HandlerLine2D(numpoints=1)})
	# plt.legend([aa, bb, cc, dd], ["Multimodal Uniform ", "Multimodal Gaussian", "Uniform", "Gaussian"],
	#            handler_map={aa: HandlerLine2D(numpoints=1), bb: HandlerLine2D(numpoints=1), cc: HandlerLine2D(numpoints=1),
	#                         dd: HandlerLine2D(numpoints=1)
	
	# }, loc='middle right')
	# plt.yticks(np.arange(0.5, 0.7, step=0.05))
	plt.yticks(np.arange(0.45, 0.65, step=0.05))
	# plt.xticks(np.arange(0, 800, step=50))
	plt.legend(loc='best')
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy Score")
	# plt.axis("auto")
	plt.grid(True)
	plt.show()
	plt.savefig(title + ".png")
	plt.close()


def MM_mu_01_zoom_plot_from_pkl_5modals():
	import matplotlib.pyplot as plt
	import pickle
	plt.Figure(figsize=(15, 15))
	
	title = 'MMinfoGAN_Fsion-Mnist_multi-modal Gaussian Sampler 5 modals mu=0.1'
	plt.title(title, fontsize=10)
	a = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.1_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	# b = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.13_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	# c = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	d = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.17_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	# e = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	f = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	
	a2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.1_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	# b2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.13_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	# c2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	d2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.17_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	# e2 = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	f2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	
	a3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.1_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	# b3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.13_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	# c3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	d3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.17_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	# e3 = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	f3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	
	a_range = np.arange(START, len(a) + START)
	# b_range = np.arange(START, len(b) + START)
	# c_range = np.arange(START, len(c) + START)
	d_range = np.arange(START, len(d) + START)
	# e_range = np.arange(START, len(e) + START)
	f_range = np.arange(START, len(f) + START)
	# g_range = np.arange(START, len(g) + START)
	
	# CALC MEAN AND STDERR
	a_mean = np.mean([a, a2, a3], axis=0)
	a_stderr = np.std([a, a2, a3], axis=0) / np.sqrt(len(a))
	# b_mean = np.mean([b, b2], axis=0)
	# b_stderr = np.std([b, b2], axis=0) / np.sqrt(len(a))
	# c_mean = np.mean([c, c2], axis=0)
	# c_stderr = np.std([c, c2], axis=0) / np.sqrt(len(c))
	d_mean = np.mean([d, d2, d3], axis=0)
	d_stderr = np.std([d, d2, d3], axis=0) / np.sqrt(len(d))
	# e_mean = np.mean([e, e2], axis=0)
	# e_stderr = np.std([e, e2], axis=0) / np.sqrt(len(e))
	f_mean = np.mean([f, f2, f3], axis=0)
	f_stderr = np.std([f, f2, f3], axis=0) / np.sqrt(len(f))
	# g_mean = np.mean([g, g2, g3], axis=0)
	# g_stderr = np.std([g, g2, g3], axis=0) / np.sqrt(len(g))
	plt.errorbar(a_range, a_mean, yerr=a_stderr, color='r', ls='--', label="$\Sigma=0.1,\mu=0.1$", marker='+', capsize=5, capthick=1, ecolor='black')
	# plt.errorbar(b_range, b_mean, yerr=b_stderr, color='r', ls='--', label="$\Sigma=0.13,\mu=0.1$",marker='.', capsize=5, capthick=1, ecolor='black')
	# plt.errorbar(c_range, c_mean, yerr=c_stderr, color='g', marker='.', label="$\Sigma=0.15,\mu=0.1$", ls='--', capsize=5, capthick=1, ecolor='k')
	plt.errorbar(d_range, d_mean, yerr=d_stderr, color='g', marker='+', label="$\Sigma=0.17,\mu=0.1$", ls='--', capsize=5, capthick=1, ecolor='k')
	# plt.errorbar(e_range, e_mean, yerr=e_stderr, color='b', ls='--', marker='.', capsize=5, capthick=1, ecolor='black')
	plt.errorbar(f_range, f_mean, yerr=f_stderr, color='c', marker='.', label="$\Sigma=0.25,\mu=0.1$", ls='--', capsize=5, capthick=1, ecolor='k')
	# plt.errorbar(g_range, g_mean, yerr=g_stderr, color='r', marker='.', label="$\Sigma=0.3,\mu=0.8$", ls='--', capsize=5, capthick=1, ecolor='k')
	
	#
	# aa, = plt.plot(a_range, a, color='b', marker="P", label="$\Sigma=0.1,\mu=0.1$", linewidth=0.5)
	# bb, = plt.plot(b_range, b, color='g', marker='d', label="$\Sigma=0.13,\mu=0.1$", linewidth=0.5)
	# cc, = plt.plot(c_range, c, color='r', marker='^', label="$\Sigma=0.15,\mu=0.1$", linewidth=0.5)
	# dd, = plt.plot(d_range, d, color='y', marker="+", label="$\Sigma=0.17,\mu=0.1$", linewidth=0.5)
	# ee, = plt.plot(e_range, e, color='k', marker="P", label="$\Sigma=0.2,\mu=0.1$", linewidth=0.5)
	# ff, = plt.plot(f_range, f, color='c', marker='.', label="$\Sigma=0.25,\mu=0.1$", linewidth=0.5)
	# gg, = plt.plot(g_range, g, color='r', marker='d', label="$\sigma=0.3,\mu=0$", linewidth=0.5)
	# hh, = plt.plot(h_range, h, color='c', marker=".", label="$\sigma=0.2,\mu=0$", linewidth=0.5)
	# ii, = plt.plot(i_range, i, color='y', marker="^", label="$\sigma=0.1,\mu=0$", linewidth=0.5)
	# jj, = plt.plot(j_range, j, color='m', marker=".", label="$\sigma=0.15,\mu=0$", linewidth=0.5)
	# kk, = plt.plot(k_range, k, color='g', marker="*", label="$\sigma=1.0,\mu=0$", linewidth=0.5)
	
	# plt.legend(handler_map={aa: HandlerLine2D(numpoints=1)})
	# plt.legend([aa, bb, cc, dd], ["Multimodal Uniform ", "Multimodal Gaussian", "Uniform", "Gaussian"],
	#            handler_map={aa: HandlerLine2D(numpoints=1), bb: HandlerLine2D(numpoints=1), cc: HandlerLine2D(numpoints=1),
	#                         dd: HandlerLine2D(numpoints=1)
	
	# }, loc='middle right')
	plt.yticks(np.arange(0.45, 0.65, step=0.05))
	plt.legend(loc='best')
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy Score")
	# plt.axis("auto")
	plt.grid(True)
	plt.show()
	plt.savefig(title + ".png")
	plt.close()


def MM_mu_1_zoom_plot_from_pkl_5modals():
	import matplotlib.pyplot as plt
	import pickle
	plt.Figure(figsize=(15, 15))
	
	title = 'MMinfoGAN_Fsion-Mnist_multi-modal Gaussian Sampler 5 modals mu=1.0'
	plt.title(title, fontsize=10)
	# a = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.1_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	b = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	# c = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.22_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	# d = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	e = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	f = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.4_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	g = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.5_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	
	# a2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.1_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	b2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	# c = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.22_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	# d = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	e2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	f2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.4_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	g2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.5_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	#
	
	# a3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.1_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	b3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	# c = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.22_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	# d = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	e3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	f3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.4_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	g3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.5_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:END]
	#
	
	# a_range = np.arange(START, len(a) + START)
	b_range = np.arange(START, len(b) + START)
	# c_range = np.arange(START, len(c) + START)
	# d_range = np.arange(START, len(d) + START)
	e_range = np.arange(START, len(e) + START)
	f_range = np.arange(START, len(f) + START)
	g_range = np.arange(START, len(g) + START)
	# CALC MEAN AND STDERR
	# a_mean = np.mean([a, a2], axis=0)
	# a_stderr = np.std([a, a2], axis=0) / np.sqrt(len(a))
	b_mean = np.mean([b, b2, b3], axis=0)
	b_stderr = np.std([b, b2, b3], axis=0) / np.sqrt(len(b))
	# c_mean = np.mean([c, c2, c3], axis=0)
	# c_stderr = np.std([c, c2, c3], axis=0) / np.sqrt(len(c))
	# d_mean = np.mean([d, d2, d3], axis=0)
	# d_stderr = np.std([d, d2, d3], axis=0) / np.sqrt(len(d))
	e_mean = np.mean([e, e2], axis=0)
	e_stderr = np.std([e, e2], axis=0) / np.sqrt(len(e))
	f_mean = np.mean([f, f2, f3], axis=0)
	f_stderr = np.std([f, f2, f3], axis=0) / np.sqrt(len(f))
	g_mean = np.mean([g, g2, g3], axis=0)
	g_stderr = np.std([g, g2, g3], axis=0) / np.sqrt(len(g))
	# plt.errorbar(a_range, a_mean, yerr=a_stderr, color='y', ls='--', marker='.', capsize=5, capthick=1, ecolor='black')
	plt.errorbar(b_range, b_mean, yerr=b_stderr, color='g', ls='--', label="$\Sigma=0.2,\mu=1.0$", marker='+', capsize=5, capthick=1, ecolor='black')
	# plt.errorbar(c_range, c_mean, yerr=c_stderr, color='g', marker='.', label="$\Sigma=0.15,\mu=0.7$", ls='--', capsize=5, capthick=1, ecolor='k')
	# plt.errorbar(d_range, d_mean, yerr=d_stderr, color='c', marker='.', label="$\Sigma=0.25,\mu=0.7$", ls='--', capsize=5, capthick=1, ecolor='k')
	plt.errorbar(e_range, e_mean, yerr=e_stderr, color='r', ls='--', label="$\Sigma=0.3,\mu=1.0$", marker='d', capsize=5, capthick=1, ecolor='black')
	# plt.errorbar(f_range, f_mean, yerr=f_stderr, color='g', marker='.', label="$\Sigma=0.4,\mu=1.0$", ls='--', capsize=5, capthick=1, ecolor='k')
	plt.errorbar(g_range, g_mean, yerr=g_stderr, color='c', marker='.', label="$\Sigma=0.5,\mu=1.0$", ls='--', capsize=5, capthick=1, ecolor='k')
	
	# aa, = plt.plot(a_range, a, color='b', marker="P", label="$\Sigma=0.15,\mu=1.0$", linewidth=0.5)
	# bb, = plt.plot(b_range, b, color='g', marker='d', label="$\Sigma=0.2,\mu=1.0$", linewidth=0.5)
	# # cc, = plt.plot(c_range, c, color='r', marker='^', label="$\sigma=0.22,\mu=1.0$", linewidth=0.5)
	# # dd, = plt.plot(d_range, d, color='y', marker=".", label="$\sigma=0.25,\mu=1.0$", linewidth=0.5)
	# ee, = plt.plot(e_range, e, color='k', marker="P", label="$\Sigma=0.3,\mu=1.0$", linewidth=0.5)
	# ff, = plt.plot(f_range, f, color='y', marker='^', label="$\Sigma=0.4,\mu=1.0$", linewidth=0.5)
	# gg, = plt.plot(g_range, g, color='r', marker='d', label="$\Sigma=0.5,\mu=1.0$", linewidth=0.5)
	# hh, = plt.plot(h_range, h, color='c', marker=".", label="$\sigma=0.2,\mu=1.0$", linewidth=0.5)
	# ii, = plt.plot(i_range, i, color='y', marker="^", label="$\sigma=0.1,\mu=0$", linewidth=0.5)
	# jj, = plt.plot(j_range, j, color='m', marker=".", label="$\sigma=0.15,\mu=0$", linewidth=0.5)
	# kk, = plt.plot(k_range, k, color='g', marker="*", label="$\sigma=1.0,\mu=0$", linewidth=0.5)
	
	# plt.legend(handler_map={aa: HandlerLine2D(numpoints=1)})
	# plt.legend([aa, bb, cc, dd], ["Multimodal Uniform ", "Multimodal Gaussian", "Uniform", "Gaussian"],
	#            handler_map={aa: HandlerLine2D(numpoints=1), bb: HandlerLine2D(numpoints=1), cc: HandlerLine2D(numpoints=1),
	#                         dd: HandlerLine2D(numpoints=1)
	
	# }, loc='middle right')
	plt.yticks(np.arange(0.45, 0.65, step=0.05))
	plt.legend(loc='best')
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy Score")
	# plt.axis("auto")
	plt.grid(True)
	plt.show()
	plt.savefig(title + ".png")
	plt.close()

#3 modalities


def MM_mu_05_07_08_zoom_plot_from_pkl3modals():
	import numpy as np
	import matplotlib.pyplot as plt
	import pickle
	plt.Figure(figsize=(15, 15))
	
	title = 'MMinfoGAN_Fsion-Mnist_multi-modal Gaussian Sampler 3 modals'
	plt.title(title, fontsize=10)
	# a = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.5_sigma_0.17_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	# b = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	c = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	# d = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	# e = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	f = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	g = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	# DIR 2
	# a2 = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.5_sigma_0.17_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	# b2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	c2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	d2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	e2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	f2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	g2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	
	# DIR 3
	# a3 = pickle.load(
	# 	open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.5_sigma_0.17_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	# b3 = pickle.load(
	# 	open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	c3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	d3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	# e3 = pickle.load(
	# 	open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	f3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	g3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	# a_range = np.arange(START, len(a) + START)
	# b_range = np.arange(START, len(b) + START)
	c_range = np.arange(START, len(c) + START)
	# d_range = np.arange(START, len(d) + START)
	# e_range = np.arange(START, len(e) + START)
	f_range = np.arange(START, len(f) + START)
	g_range = np.arange(START, len(g) + START)
	
	# CALC MEAN AND STDERR
	# a_mean = np.mean([a, a2], axis=0)
	# a_stderr = np.std([a, a2], axis=0) / np.sqrt(len(a))
	# b_mean = np.mean([b, b2], axis=0)
	# b_stderr = np.std([b, b2], axis=0) / np.sqrt(len(a))
	c_mean = np.mean([c, c2, c3], axis=0)
	c_stderr = np.std([c, c2, c3], axis=0) / np.sqrt(len(c))
	d_mean = np.mean([d, d2, d3], axis=0)
	d_stderr = np.std([d, d2, d3], axis=0) / np.sqrt(len(d))
	e_mean = np.mean([e, e2], axis=0)
	e_stderr = np.std([e, e2], axis=0) / np.sqrt(len(e))
	f_mean = np.mean([f, f2, f3], axis=0)
	f_stderr = np.std([f, f2, f3], axis=0) / np.sqrt(len(f))
	g_mean = np.mean([g, g2, g3], axis=0)
	g_stderr = np.std([g, g2, g3], axis=0) / np.sqrt(len(g))
	# plt.errorbar(a_range, a_mean, yerr=a_stderr, color='y', ls='--', marker='p', capsize=5, capthick=1, ecolor='black')
	# plt.errorbar(b_range, b_mean, yerr=b_stderr, color='red', ls='--', marker='.', capsize=5, capthick=1, ecolor='black')
	plt.errorbar(c_range, c_mean, yerr=c_stderr, color='g', marker='d', label="$\Sigma=0.15,\mu=0.7$", ls='--', capsize=5, capthick=1, ecolor='k')
	# plt.errorbar(d_range, d_mean, yerr=d_stderr, color='c', marker='.', label="$\Sigma=0.25,\mu=0.7$", ls='--', capsize=5, capthick=1, ecolor='k')
	# plt.errorbar(e_range, e_mean, yerr=e_stderr, color='b', ls='--', marker='.', capsize=5, capthick=1, ecolor='black')
	plt.errorbar(f_range, f_mean, yerr=f_stderr, color='c', marker='*', label="$\Sigma=0.2,\mu=0.8$", ls='--', capsize=5, capthick=1, ecolor='k')
	plt.errorbar(g_range, g_mean, yerr=g_stderr, color='r', marker='.', label="$\Sigma=0.3,\mu=0.8$", ls='--', capsize=5, capthick=1, ecolor='k')
	
	# aa, = plt.plot(a_range, a, color='b', marker="P", label="$\Sigma=0.17,\mu=0.5$", linewidth=0.5)
	# bb, = plt.plot(b_range, b, color='g', marker='d', label="$\Sigma=0.15,\mu=0.7$", linewidth=0.5)
	# cc, = plt.plot(c_range, c, color='r', marker='^', label="$\Sigma=0.25,\mu=0.7$", linewidth=0.5)
	# dd, = plt.plot(d_range, d, color='r', marker=".", label="$\Sigma=0.3,\mu=0.7$", linewidth=0.5)
	# ee, = plt.plot(e_range, e, color='k', marker="P", label="$\Sigma=0.15,\mu=0.8$", linewidth=0.5)
	# ff, = plt.plot(f_range, f, color='b', marker='+', label="$\Sigma=0.2,\mu=0.8$", linewidth=0.5)
	# gg, = plt.plot(g_range, g, color='r', marker='d', label="$\Sigma=0.3,\mu=08$", linewidth=0.5)
	# hh, = plt.plot(h_range, h, color='c', marker=".", label="$\sigma=0.8,\mu=0.8$", linewidth=0.5)
	# ii, = plt.plot(i_range, i, color='y', marker="^", label="$\sigma=0.1,\mu=0$", linewidth=0.5)
	# jj, = plt.plot(j_range, j, color='m', marker=".", label="$\sigma=0.15,\mu=0$", linewidth=0.5)
	# kk, = plt.plot(k_range, k, color='g', marker="*", label="$\sigma=1.0,\mu=0$", linewidth=0.5)
	
	# plt.legend(handler_map={aa: HandlerLine2D(numpoints=1)})
	# plt.legend([aa, bb, cc, dd], ["Multimodal Uniform ", "Multimodal Gaussian", "Uniform", "Gaussian"],
	#            handler_map={aa: HandlerLine2D(numpoints=1), bb: HandlerLine2D(numpoints=1), cc: HandlerLine2D(numpoints=1),
	#                         dd: HandlerLine2D(numpoints=1)
	
	# }, loc='middle right')
	# plt.yticks(np.arange(0.5, 0.7, step=0.05))
	plt.yticks(np.arange(0.45, 0.65, step=0.05))
	# plt.xticks(np.arange(0, 800, step=50))
	plt.legend(loc='best')
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy Score")
	# plt.axis("auto")
	plt.grid(True)
	plt.show()
	plt.savefig(title + ".png")
	plt.close()


def MM_mu_01_zoom_plot_from_pkl3modals():
	import matplotlib.pyplot as plt
	import pickle
	plt.Figure(figsize=(15, 15))
	
	title = 'MMinfoGAN_Fsion-Mnist_multi-modal Gaussian Sampler 3 modals mu=0.1'
	plt.title(title, fontsize=10)
	a = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.1_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	# b = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.13_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	# c = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	d = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.17_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	# e = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	f = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	
	a2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.1_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	# b2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.13_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	# c2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	d2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.17_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	# e2 = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	f2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	
	a3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.1_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	# b3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.13_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	# c3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	d3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.17_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	# e3 = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	f3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	
	a_range = np.arange(START, len(a) + START)
	# b_range = np.arange(START, len(b) + START)
	# c_range = np.arange(START, len(c) + START)
	d_range = np.arange(START, len(d) + START)
	# e_range = np.arange(START, len(e) + START)
	f_range = np.arange(START, len(f) + START)
	# g_range = np.arange(START, len(g) + START)
	
	# CALC MEAN AND STDERR
	a_mean = np.mean([a, a2, a3], axis=0)
	a_stderr = np.std([a, a2, a3], axis=0) / np.sqrt(len(a))
	# b_mean = np.mean([b, b2], axis=0)
	# b_stderr = np.std([b, b2], axis=0) / np.sqrt(len(a))
	# c_mean = np.mean([c, c2], axis=0)
	# c_stderr = np.std([c, c2], axis=0) / np.sqrt(len(c))
	d_mean = np.mean([d, d2, d3], axis=0)
	d_stderr = np.std([d, d2, d3], axis=0) / np.sqrt(len(d))
	# e_mean = np.mean([e, e2], axis=0)
	# e_stderr = np.std([e, e2], axis=0) / np.sqrt(len(e))
	f_mean = np.mean([f, f2, f3], axis=0)
	f_stderr = np.std([f, f2, f3], axis=0) / np.sqrt(len(f))
	# g_mean = np.mean([g, g2, g3], axis=0)
	# g_stderr = np.std([g, g2, g3], axis=0) / np.sqrt(len(g))
	plt.errorbar(a_range, a_mean, yerr=a_stderr, color='r', ls='--', label="$\Sigma=0.1,\mu=0.1$", marker='+', capsize=5, capthick=1, ecolor='black')
	# plt.errorbar(b_range, b_mean, yerr=b_stderr, color='r', ls='--', label="$\Sigma=0.13,\mu=0.1$",marker='.', capsize=5, capthick=1, ecolor='black')
	# plt.errorbar(c_range, c_mean, yerr=c_stderr, color='g', marker='.', label="$\Sigma=0.15,\mu=0.1$", ls='--', capsize=5, capthick=1, ecolor='k')
	plt.errorbar(d_range, d_mean, yerr=d_stderr, color='g', marker='+', label="$\Sigma=0.17,\mu=0.1$", ls='--', capsize=5, capthick=1, ecolor='k')
	# plt.errorbar(e_range, e_mean, yerr=e_stderr, color='b', ls='--', marker='.', capsize=5, capthick=1, ecolor='black')
	plt.errorbar(f_range, f_mean, yerr=f_stderr, color='c', marker='.', label="$\Sigma=0.25,\mu=0.1$", ls='--', capsize=5, capthick=1, ecolor='k')
	# plt.errorbar(g_range, g_mean, yerr=g_stderr, color='r', marker='.', label="$\Sigma=0.3,\mu=0.8$", ls='--', capsize=5, capthick=1, ecolor='k')
	
	#
	# aa, = plt.plot(a_range, a, color='b', marker="P", label="$\Sigma=0.1,\mu=0.1$", linewidth=0.5)
	# bb, = plt.plot(b_range, b, color='g', marker='d', label="$\Sigma=0.13,\mu=0.1$", linewidth=0.5)
	# cc, = plt.plot(c_range, c, color='r', marker='^', label="$\Sigma=0.15,\mu=0.1$", linewidth=0.5)
	# dd, = plt.plot(d_range, d, color='y', marker="+", label="$\Sigma=0.17,\mu=0.1$", linewidth=0.5)
	# ee, = plt.plot(e_range, e, color='k', marker="P", label="$\Sigma=0.2,\mu=0.1$", linewidth=0.5)
	# ff, = plt.plot(f_range, f, color='c', marker='.', label="$\Sigma=0.25,\mu=0.1$", linewidth=0.5)
	# gg, = plt.plot(g_range, g, color='r', marker='d', label="$\sigma=0.3,\mu=0$", linewidth=0.5)
	# hh, = plt.plot(h_range, h, color='c', marker=".", label="$\sigma=0.2,\mu=0$", linewidth=0.5)
	# ii, = plt.plot(i_range, i, color='y', marker="^", label="$\sigma=0.1,\mu=0$", linewidth=0.5)
	# jj, = plt.plot(j_range, j, color='m', marker=".", label="$\sigma=0.15,\mu=0$", linewidth=0.5)
	# kk, = plt.plot(k_range, k, color='g', marker="*", label="$\sigma=1.0,\mu=0$", linewidth=0.5)
	
	# plt.legend(handler_map={aa: HandlerLine2D(numpoints=1)})
	# plt.legend([aa, bb, cc, dd], ["Multimodal Uniform ", "Multimodal Gaussian", "Uniform", "Gaussian"],
	#            handler_map={aa: HandlerLine2D(numpoints=1), bb: HandlerLine2D(numpoints=1), cc: HandlerLine2D(numpoints=1),
	#                         dd: HandlerLine2D(numpoints=1)
	
	# }, loc='middle right')
	plt.yticks(np.arange(0.45, 0.65, step=0.05))
	plt.legend(loc='best')
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy Score")
	# plt.axis("auto")
	plt.grid(True)
	plt.show()
	plt.savefig(title + ".png")
	plt.close()


def MM_mu_1_zoom_plot_from_pkl3modals():
	import matplotlib.pyplot as plt
	import pickle
	plt.Figure(figsize=(15, 15))
	
	title = 'MMinfoGAN_Fsion-Mnist_multi-modal Gaussian Sampler 3 modals mu=1.0'
	plt.title(title, fontsize=10)
	# a = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.1_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:END]
	b = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	# c = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.22_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	# d = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	e = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	# f = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.4_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	g = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.5_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	
	# a2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.1_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	b2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	# c = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.22_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	# d = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	e2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	f2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.4_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	g2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.5_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	#
	
	# a3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.1_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	b3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	# c = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.22_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	# d = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	e3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	f3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.4_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	g3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.5_czcc_czrc_rzcc_rzrc_ndist_3_accuracy.pkl", "rb"))[START:END]
	#
	
	# a_range = np.arange(START, len(a) + START)
	b_range = np.arange(START, len(b) + START)
	# c_range = np.arange(START, len(c) + START)
	# d_range = np.arange(START, len(d) + START)
	e_range = np.arange(START, len(e) + START)
	f_range = np.arange(START, len(f2) + START)
	g_range = np.arange(START, len(g) + START)
	# CALC MEAN AND STDERR
	# a_mean = np.mean([a, a2], axis=0)
	# a_stderr = np.std([a, a2], axis=0) / np.sqrt(len(a))
	b_mean = np.mean([b, b2, b3], axis=0)
	b_stderr = np.std([b, b2, b3], axis=0) / np.sqrt(len(b))
	# c_mean = np.mean([c, c2, c3], axis=0)
	# c_stderr = np.std([c, c2, c3], axis=0) / np.sqrt(len(c))
	# d_mean = np.mean([d, d2, d3], axis=0)
	# d_stderr = np.std([d, d2, d3], axis=0) / np.sqrt(len(d))
	e_mean = np.mean([e, e2], axis=0)
	e_stderr = np.std([e, e2], axis=0) / np.sqrt(len(e))
	f_mean = np.mean([f2, f3], axis=0)
	f_stderr = np.std([f2, f3], axis=0) / np.sqrt(len(f2))
	g_mean = np.mean([g, g2, g3], axis=0)
	g_stderr = np.std([g, g2, g3], axis=0) / np.sqrt(len(g))
	# plt.errorbar(a_range, a_mean, yerr=a_stderr, color='y', ls='--', marker='.', capsize=5, capthick=1, ecolor='black')
	plt.errorbar(b_range, b_mean, yerr=b_stderr, color='g', ls='--', label="$\Sigma=0.2,\mu=1.0$", marker='+', capsize=5, capthick=1, ecolor='black')
	# plt.errorbar(c_range, c_mean, yerr=c_stderr, color='g', marker='.', label="$\Sigma=0.15,\mu=0.7$", ls='--', capsize=5, capthick=1, ecolor='k')
	# plt.errorbar(d_range, d_mean, yerr=d_stderr, color='c', marker='.', label="$\Sigma=0.25,\mu=0.7$", ls='--', capsize=5, capthick=1, ecolor='k')
	plt.errorbar(e_range, e_mean, yerr=e_stderr, color='r', ls='--', label="$\Sigma=0.3,\mu=1.0$", marker='d', capsize=5, capthick=1, ecolor='black')
	# plt.errorbar(f_range, f_mean, yerr=f_stderr, color='g', marker='.', label="$\Sigma=0.4,\mu=1.0$", ls='--', capsize=5, capthick=1, ecolor='k')
	plt.errorbar(g_range, g_mean, yerr=g_stderr, color='c', marker='.', label="$\Sigma=0.5,\mu=1.0$", ls='--', capsize=5, capthick=1, ecolor='k')
	
	# aa, = plt.plot(a_range, a, color='b', marker="P", label="$\Sigma=0.15,\mu=1.0$", linewidth=0.5)
	# bb, = plt.plot(b_range, b, color='g', marker='d', label="$\Sigma=0.2,\mu=1.0$", linewidth=0.5)
	# # cc, = plt.plot(c_range, c, color='r', marker='^', label="$\sigma=0.22,\mu=1.0$", linewidth=0.5)
	# # dd, = plt.plot(d_range, d, color='y', marker=".", label="$\sigma=0.25,\mu=1.0$", linewidth=0.5)
	# ee, = plt.plot(e_range, e, color='k', marker="P", label="$\Sigma=0.3,\mu=1.0$", linewidth=0.5)
	# ff, = plt.plot(f_range, f, color='y', marker='^', label="$\Sigma=0.4,\mu=1.0$", linewidth=0.5)
	# gg, = plt.plot(g_range, g, color='r', marker='d', label="$\Sigma=0.5,\mu=1.0$", linewidth=0.5)
	# hh, = plt.plot(h_range, h, color='c', marker=".", label="$\sigma=0.2,\mu=1.0$", linewidth=0.5)
	# ii, = plt.plot(i_range, i, color='y', marker="^", label="$\sigma=0.1,\mu=0$", linewidth=0.5)
	# jj, = plt.plot(j_range, j, color='m', marker=".", label="$\sigma=0.15,\mu=0$", linewidth=0.5)
	# kk, = plt.plot(k_range, k, color='g', marker="*", label="$\sigma=1.0,\mu=0$", linewidth=0.5)
	
	# plt.legend(handler_map={aa: HandlerLine2D(numpoints=1)})
	# plt.legend([aa, bb, cc, dd], ["Multimodal Uniform ", "Multimodal Gaussian", "Uniform", "Gaussian"],
	#            handler_map={aa: HandlerLine2D(numpoints=1), bb: HandlerLine2D(numpoints=1), cc: HandlerLine2D(numpoints=1),
	#                         dd: HandlerLine2D(numpoints=1)
	
	# }, loc='middle right')
	plt.yticks(np.arange(0.45, 0.65, step=0.05))
	plt.legend(loc='best')
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy Score")
	# plt.axis("auto")
	plt.grid(True)
	plt.show()
	plt.savefig(title + ".png")
	plt.close()
if __name__ == '__main__':
	# fashion_MM_plot_from_pkl()
	# plot_from_pkl()
	# MM_mu_1_zoom_plot_from_pkl()
	# MM_mu_01_zoom_plot_from_pkl()
	# MM_mu_05_07_08_zoom_plot_from_pkl()
	# MM_mu_05_07_08_zoom_plot_from_pkl_5_modals()
	# MM_mu_1_zoom_plot_from_pkl_5modals()
	# MM_mu_01_zoom_plot_from_pkl_5modals()
	# MM_mu_1_zoom_plot_from_pkl3modals()
	# MM_mu_01_zoom_plot_from_pkl3modals()
	MM_mu_05_07_08_zoom_plot_from_pkl3modals()