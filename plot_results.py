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
START = 0
start=START

def plot_from_pkl():
	plt.Figure(figsize=(15, 15))
	
	plt.title('MMinfoGAN Mnist Different Priros Accuracy', fontsize=12)
	e = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_GaussianSample_mu_0.0_sigma_0.2_czcc_czrc_rzcc_rzrc_accuracy.pkl", "rb"))[start:50]
	f = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.2_czcc_czrc_rzcc_rzrc_accuracy.pkl", "rb"))[start:50]
	g = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_UniformSample_mu_0_sigma_0.15_czcc_czrc_rzcc_rzrc_accuracy.pkl", "rb"))[start:50]
	h = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_MultiModalUniformSample_mu_0_sigma_0.15_czcc_czrc_rzcc_rzrc_accuracy.pkl", "rb"))[start:50]
	e_range = np.arange(len(e))
	f_range = np.arange(len(f))
	g_range = np.arange(len(g))
	h_range = np.arange(len(h))
	# i_range = np.arange(len(i))
	# j_range = np.arange(len(j))
	ee, = plt.plot(e_range, e, color='k', marker="P", label="Multi-modal uniform", linewidth=0.5)
	ff, = plt.plot(f_range, f, color='b', marker='.', label="$\mathbb{N}(\sigma=0.2,\mu=0$)", linewidth=0.5)
	gg, = plt.plot(g_range, g, color='r', marker='d', label="Uniform", linewidth=0.5)
	hh, = plt.plot(h_range, h, color='c', marker=".", label="$\mathbb{N}(\Sigma=0.2,\mu=0.1$)", linewidth=0.5)
	# jj, = plt.plot(j_range, j, color='m', marker=".", label="$\sigma=0.1,\mu=0$", linewidth=0.5)
	# ii, = plt.plot(i_range, i, color='y', marker="^", label="$\sigma=0.15,\mu=0$", linewidth=0.5)
	# mean_line = plt.plot(e_range, np.ones_like(e_range) * 0.92, label='Benchmark', linestyle='--')
	
	# plt.legend(handler_map={aa: HandlerLine2D(numpoints=1)})
	# plt.legend([ee, ff, gg], ["Gaussian", "Multimodal Gaussian", "Uniform"],
	#            handler_map={ee: HandlerLine2D(numpoints=1), ff: HandlerLine2D(numpoints=1), gg: HandlerLine2D(numpoints=1)
	# 	 }, loc='middle right')
	plt.legend(loc='best')
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy Score")
	# plt.axis("auto")
	plt.grid(True)
	plt.show()
	plt.savefig("MMinfoGAN_mnist_MultivariateGaussianSampler.png")
	plt.close()


def fashion_MM_plot_from_pkl():
	import numpy as np
	import matplotlib.pyplot as plt
	import pickle
	plt.Figure(figsize=(15, 15))
	
	plt.title('MMinfoGAN Fashion-Mnist Different Priros Accuracy', fontsize=12)
	a = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_GaussianSample_mu_0.0_sigma_0.2_czcc_czrc_rzcc_rzrc_accuracy.pkl", "rb"))[START:50]
	b = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:50]
	c = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_UniformSample_mu_0.0_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:50]
	d = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultiModalUniformSample_mu_0.0_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:50]
	
	a2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_GaussianSample_mu_0.0_sigma_0.2_czcc_czrc_rzcc_rzrc_accuracy.pkl", "rb"))[START:50]
	b2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:50]
	c2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_UniformSample_mu_0.0_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:50]
	d2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultiModalUniformSample_mu_0.0_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:50]
	
	# DIR 3
	a3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_GaussianSample_mu_0.0_sigma_0.2_czcc_czrc_rzcc_rzrc_accuracy.pkl", "rb"))[START:50]
	b3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_5_accuracy.pkl", "rb"))[START:50]
	c3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_UniformSample_mu_0.0_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:50]
	d3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultiModalUniformSample_mu_0.0_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:50]
	
	
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
	plt.errorbar(a_range, a_mean, yerr=a_stderr, color='m', ls='--', marker='o', capsize=5, capthick=1, ecolor='k')
	plt.errorbar(b_range, b_mean, yerr=b_stderr, color='y', marker='+', label="$\Sigma=0.3,\mu=0.7$", ls='--', capsize=5, capthick=1, ecolor='k')
	plt.errorbar(c_range, c_mean, yerr=c_stderr, color='g', marker='d', label="uniform", ls='--', capsize=5, capthick=1, ecolor='k')
	plt.errorbar(d_range, d_mean, yerr=d_stderr, color='c', marker='^', label="multi-modal unifrom", ls='--', capsize=5, capthick=1, ecolor='k')
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
	plt.legend(loc='best')
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy Score")
	# plt.axis("auto")
	plt.grid(True)
	plt.show()
	plt.savefig("MMinfoGAN_mnist_MultivariateGaussianSampler.png")
	plt.close()


# ACCURACY
def truncated__zoom_plot_from_pkl():
	import numpy as np
	import matplotlib.pyplot as plt
	import pickle
	plt.Figure(figsize=(15, 15))
	
	plt.title('Zoom_MMinfoGAN_mnist_TruncatedGaussianSample', fontsize=12)
	a = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_TruncatedGaussianSample_mu_0.0_sigma_10.0_accuracy.pkl", "rb"))[2:]
	b = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_TruncatedGaussianSample_mu_0.0_sigma_5.0_accuracy.pkl", "rb"))[2:]
	c = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_TruncatedGaussianSample_mu_0.0_sigma_3.0_accuracy.pkl", "rb"))[2:]
	d = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_TruncatedGaussianSample_mu_0.0_sigma_2.0_accuracy.pkl", "rb"))[2:]
	e = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_TruncatedGaussianSample_mu_0.0_sigma_0.5_accuracy.pkl", "rb"))[2:]
	f = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_TruncatedGaussianSample_mu_0.0_sigma_0.4_accuracy.pkl", "rb"))[2:]
	g = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_TruncatedGaussianSample_mu_0.0_sigma_0.3_accuracy.pkl", "rb"))[2:]
	h = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_TruncatedGaussianSample_mu_0.0_sigma_0.2_accuracy.pkl", "rb"))[2:]
	# i= pickle.load(open(dir+"classifier_MMinfoGAN_mnist_TruncatedGaussianSample_mu_0.0_sigma_0.1_accuracy.pkl", "rb"))
	j = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_TruncatedGaussianSample_mu_0.0_sigma_0.15_accuracy.pkl", "rb"))[2:]
	k = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_TruncatedGaussianSample_mu_0.0_sigma_1.0_accuracy.pkl", "rb"))[2:]
	# plt.plot(a, np.arange(len(a)), 'r--',  b,np.arange(len(b)), 'b--',  c,np.arange(len(c)),'g^',d,np.arange(len(d)),"y--")
	a_range = np.arange(2, len(a) + 2)
	b_range = np.arange(2, len(b) + 2)
	c_range = np.arange(2, len(c) + 2)
	d_range = np.arange(2, len(d) + 2)
	e_range = np.arange(2, len(e) + 2)
	f_range = np.arange(2, len(f) + 2)
	g_range = np.arange(2, len(g) + 2)
	h_range = np.arange(2, len(h) + 2)
	# i_range = np.arange(len(i))
	j_range = np.arange(len(j))
	k_range = np.arange(len(k))
	aa, = plt.plot(a_range, a, color='b', marker="P", label="$\sigma=10,\mu=0$", linewidth=0.5)
	bb, = plt.plot(b_range, b, color='g', marker='d', label="$\sigma=5,\mu=0$", linewidth=0.5)
	# cc, = plt.plot(c_range, c, color='r', marker='^', label="$\sigma=3,\mu=0$", linewidth=0.5)
	dd, = plt.plot(d_range, d, color='y', marker=".", label="$\sigma=2,\mu=0$", linewidth=0.5)
	ee, = plt.plot(e_range, e, color='k', marker="P", label="$\sigma=0.5,\mu=0$", linewidth=0.5)
	# ff, = plt.plot(f_range, f, color='g', marker='.', label="$\sigma=0.4,\mu=0$", linewidth=0.5)
	# gg, = plt.plot(g_range, g, color='r', marker='>', label="$\sigma=0.3,\mu=0$", linewidth=0.5)
	# hh, = plt.plot(h_range, h, color='c', marker=".", label="$\sigma=0.2,\mu=0$", linewidth=0.5)
	# jj, = plt.plot(j_range, j, color='m', marker=".", label="$\sigma=0.15,\mu=0$", linewidth=0.5)
	# kk, = plt.plot(k_range, k, color='k', marker="*", label="$\sigma=1.0,\mu=0$", linewidth=0.5)
	mean_line = plt.plot(c_range, np.ones_like(d_range) * 0.98, label='Benchmark', linestyle='--')
	
	# plt.legend(handler_map={aa: HandlerLine2D(numpoints=1)})
	# plt.legend([aa, bb, cc, dd], ["Multimodal Uniform ", "Multimodal Gaussian", "Uniform", "Gaussian"],
	#            handler_map={aa: HandlerLine2D(numpoints=1), bb: HandlerLine2D(numpoints=1), cc: HandlerLine2D(numpoints=1),
	#                         dd: HandlerLine2D(numpoints=1)
	
	# }, loc='middle right')
	plt.legend(loc='best')
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy Score")
	# plt.axis("auto")
	plt.grid(True)
	plt.show()
	plt.savefig("Zoom_MMinfoGAN_mnist_TruncatedGaussianSample.png")
	plt.close()


def truncated_plot_from_pkl():
	import numpy as np
	import matplotlib.pyplot as plt
	import pickle
	plt.Figure(figsize=(15, 15))
	
	plt.title('MMinfoGAN_mnist_TruncatedGaussianSample', fontsize=12)
	a = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_TruncatedGaussianSample_mu_0.0_sigma_10.0_accuracy.pkl", "rb"))
	b = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_TruncatedGaussianSample_mu_0.0_sigma_5.0_accuracy.pkl", "rb"))
	c = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_TruncatedGaussianSample_mu_0.0_sigma_3.0_accuracy.pkl", "rb"))
	d = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_TruncatedGaussianSample_mu_0.0_sigma_2.0_accuracy.pkl", "rb"))
	e = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_TruncatedGaussianSample_mu_0.0_sigma_0.5_accuracy.pkl", "rb"))
	f = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_TruncatedGaussianSample_mu_0.0_sigma_0.4_accuracy.pkl", "rb"))
	g = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_TruncatedGaussianSample_mu_0.0_sigma_0.3_accuracy.pkl", "rb"))
	h = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_TruncatedGaussianSample_mu_0.0_sigma_0.2_accuracy.pkl", "rb"))
	# i= pickle.load(open(dir+"classifier_MMinfoGAN_mnist_TruncatedGaussianSample_mu_0.0_sigma_0.1_accuracy.pkl", "rb"))
	j = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_TruncatedGaussianSample_mu_0.0_sigma_0.15_accuracy.pkl", "rb"))
	k = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_TruncatedGaussianSample_mu_0.0_sigma_1.0_accuracy.pkl", "rb"))
	# plt.plot(a, np.arange(len(a)), 'r--',  b,np.arange(len(b)), 'b--',  c,np.arange(len(c)),'g^',d,np.arange(len(d)),"y--")
	a_range = np.arange(len(a))
	b_range = np.arange(len(b))
	c_range = np.arange(len(c))
	d_range = np.arange(len(d))
	e_range = np.arange(len(e))
	f_range = np.arange(len(f))
	g_range = np.arange(len(g))
	h_range = np.arange(len(h))
	# i_range = np.arange(len(i))
	j_range = np.arange(len(j))
	k_range = np.arange(len(k))
	aa, = plt.plot(a_range, a, color='b', marker="P", label="$\sigma=10,\mu=0$", linewidth=0.5)
	# bb, = plt.plot(b_range, b, color='g', marker='d', label="$\sigma=5,\mu=0$", linewidth=0.5)
	# cc, = plt.plot(c_range, c, color='r', marker='^', label="$\sigma=3,\mu=0$", linewidth=0.5)
	dd, = plt.plot(d_range, d, color='y', marker=".", label="$\sigma=2,\mu=0$", linewidth=0.5)
	ee, = plt.plot(e_range, e, color='k', marker="P", label="$\sigma=0.5,\mu=0$", linewidth=0.5)
	ff, = plt.plot(f_range, f, color='g', marker='.', label="$\sigma=0.4,\mu=0$", linewidth=0.5)
	gg, = plt.plot(g_range, g, color='c', marker='>', label="$\sigma=0.3,\mu=0$", linewidth=0.5)
	# hh, = plt.plot(h_range, h, color='c', marker=".", label="$\sigma=0.2,\mu=0$", linewidth=0.5)
	# jj, = plt.plot(j_range, j, color='m', marker=".", label="$\sigma=0.15,\mu=0$", linewidth=0.5)
	kk, = plt.plot(k_range, k, color='g', marker="*", label="$\sigma=1.0,\mu=0$", linewidth=0.5)
	mean_line = plt.plot(c_range, np.ones_like(d_range) * 0.98, label='Benchmark', linestyle='--')
	
	# plt.legend(handler_map={aa: HandlerLine2D(numpoints=1)})
	# plt.legend([aa, bb, cc, dd], ["Multimodal Uniform ", "Multimodal Gaussian", "Uniform", "Gaussian"],
	#            handler_map={aa: HandlerLine2D(numpoints=1), bb: HandlerLine2D(numpoints=1), cc: HandlerLine2D(numpoints=1),
	#                         dd: HandlerLine2D(numpoints=1)
	
	# }, loc='middle right')
	plt.legend(loc='best')
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy Score")
	# plt.axis("auto")
	plt.grid(True)
	plt.show()
	plt.savefig("MMinfoGAN_mnist_TruncatedGaussianSample.png")
	plt.close()


def gaussian_zoom_plot_from_pkl():
	import numpy as np
	import matplotlib.pyplot as plt
	import pickle
	plt.Figure(figsize=(15, 15))
	
	plt.title('Zoom_MMinfoGAN_mnist_GaussianSample', fontsize=12)
	# a= pickle.load(open(dir+"classifier_MMinfoGAN_mnist_MultivariateGaussianSample_mu_0.0_sigma_10.0_accuracy.pkl", "rb"))[2:]
	# b= pickle.load(open(dir+"classifier_MMinfoGAN_mnist_MultivariateGaussianSample_mu_0.0_sigma_5.0_accuracy.pkl", "rb"))[2:]
	# c= pickle.load(open(dir+"classifier_MMinfoGAN_mnist_MultivariateGaussianSample_mu_0.0_sigma_3.0_accuracy.pkl", "rb"))[2:]
	# d= pickle.load(open(dir+"classifier_MMinfoGAN_mnist_MultivariateGaussianSample_mu_0.0_sigma_2.0_accuracy.pkl", "rb"))[2:]
	e = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_GaussianSample_mu_0.0_sigma_0.5_accuracy.pkl", "rb"))[2:]
	f = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_GaussianSample_mu_0.0_sigma_0.4_accuracy.pkl", "rb"))[2:]
	g = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_GaussianSample_mu_0.0_sigma_0.3_accuracy.pkl", "rb"))[2:]
	h = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_GaussianSample_mu_0.0_sigma_0.2_accuracy.pkl", "rb"))[2:]
	i = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_GaussianSample_mu_0.0_sigma_0.1_accuracy.pkl", "rb"))[2:]
	j = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_GaussianSample_mu_0.0_sigma_0.15_accuracy.pkl", "rb"))[2:]
	k = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_GaussianSample_mu_0.0_sigma_1.0_accuracy.pkl", "rb"))[2:]
	# plt.plot(a, np.arange(len(a)), 'r--',  b,np.arange(len(b)), 'b--',  c,np.arange(len(c)),'g^',d,np.arange(len(d)),"y--")
	# a_range = np.arange(2,len(a)+2)
	# b_range = np.arange(2,len(b)+2)
	# c_range = np.arange(2,len(c)+2)
	# d_range = np.arange(2,len(d)+2)
	e_range = np.arange(2, len(e) + 2)
	f_range = np.arange(2, len(f) + 2)
	g_range = np.arange(2, len(g) + 2)
	h_range = np.arange(2, len(h) + 2)
	i_range = np.arange(len(i))
	j_range = np.arange(len(j))
	k_range = np.arange(len(k))
	# aa, = plt.plot(a_range, a, color='b', marker="P", label="$\sigma=10,\mu=0$", linewidth=0.5)
	# bb, = plt.plot(b_range, b, color='g', marker='d', label="$\sigma=5,\mu=0$", linewidth=0.5)
	# # cc, = plt.plot(c_range, c, color='r', marker='^', label="$\sigma=3,\mu=0$", linewidth=0.5)
	# dd, = plt.plot(d_range, d, color='y', marker=".", label="$\sigma=2,\mu=0$", linewidth=0.5)
	ee, = plt.plot(e_range, e, color='k', marker="P", label="$\sigma=0.5,\mu=0$", linewidth=0.5)
	ff, = plt.plot(f_range, f, color='b', marker='.', label="$\sigma=0.4,\mu=0$", linewidth=0.5)
	# gg, = plt.plot(g_range, g, color='r', marker='d', label="$\sigma=0.3,\mu=0$", linewidth=0.5)
	hh, = plt.plot(h_range, h, color='c', marker=".", label="$\sigma=0.2,\mu=0$", linewidth=0.5)
	ii, = plt.plot(j_range, i, color='y', marker="^", label="$\sigma=0.1,\mu=0$", linewidth=0.5)
	jj, = plt.plot(j_range, j, color='m', marker=".", label="$\sigma=0.15,\mu=0$", linewidth=0.5)
	kk, = plt.plot(k_range, k, color='g', marker="*", label="$\sigma=1.0,\mu=0$", linewidth=0.5)
	
	# plt.legend(handler_map={aa: HandlerLine2D(numpoints=1)})
	# plt.legend([aa, bb, cc, dd], ["Multimodal Uniform ", "Multimodal Gaussian", "Uniform", "Gaussian"],
	#            handler_map={aa: HandlerLine2D(numpoints=1), bb: HandlerLine2D(numpoints=1), cc: HandlerLine2D(numpoints=1),
	#                         dd: HandlerLine2D(numpoints=1)
	
	# }, loc='middle right')
	plt.legend(loc='best')
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy Score")
	# plt.axis("auto")
	plt.grid(True)
	plt.show()
	plt.savefig("Zoom_MMinfoGAN_mnist_GaussianSample.png")
	plt.close()


def gaussian_plot_from_pkl():
	import numpy as np
	import matplotlib.pyplot as plt
	import pickle
	plt.Figure(figsize=(15, 15))
	
	plt.title('MMinfoGAN_mnist_GaussianSample', fontsize=12)
	# a= pickle.load(open(dir+"classifier_MMinfoGAN_mnist_MultivariateGaussianSample_mu_0.0_sigma_10.0_accuracy.pkl", "rb"))
	# b= pickle.load(open(dir+"classifier_MMinfoGAN_mnist_MultivariateGaussianSample_mu_0.0_sigma_5.0_accuracy.pkl", "rb"))
	# c= pickle.load(open(dir+"classifier_MMinfoGAN_mnist_MultivariateGaussianSample_mu_0.0_sigma_3.0_accuracy.pkl", "rb"))
	# d= pickle.load(open(dir+"classifier_MMinfoGAN_mnist_MultivariateGaussianSample_mu_0.0_sigma_2.0_accuracy.pkl", "rb"))
	e = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_GaussianSample_mu_0.0_sigma_0.5_accuracy.pkl", "rb"))
	f = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_GaussianSample_mu_0.0_sigma_0.4_accuracy.pkl", "rb"))
	g = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_GaussianSample_mu_0.0_sigma_0.3_accuracy.pkl", "rb"))
	h = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_GaussianSample_mu_0.0_sigma_0.2_accuracy.pkl", "rb"))
	i = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_GaussianSample_mu_0.0_sigma_0.1_accuracy.pkl", "rb"))
	j = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_GaussianSample_mu_0.0_sigma_0.15_accuracy.pkl", "rb"))
	k = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_GaussianSample_mu_0.0_sigma_1.0_accuracy.pkl", "rb"))
	# a_range = np.arange(len(a))
	# b_range = np.arange(len(b))
	# c_range = np.arange(len(c))
	# d_range = np.arange(len(d))
	e_range = np.arange(len(e))
	f_range = np.arange(len(f))
	g_range = np.arange(len(g))
	h_range = np.arange(len(h))
	i_range = np.arange(len(i))
	j_range = np.arange(len(j))
	k_range = np.arange(len(k))
	# aa, = plt.plot(a_range, a, color='b', marker="P", label="$\sigma=10,\mu=0$", linewidth=0.5)
	# bb, = plt.plot(b_range, b, color='g', marker='d', label="$\sigma=5,\mu=0$", linewidth=0.5)
	# cc, = plt.plot(c_range, c, color='r', marker='^', label="$\sigma=3,\mu=0$", linewidth=0.5)
	# dd, = plt.plot(d_range, d, color='y', marker=".", label="$\sigma=2,\mu=0$", linewidth=0.5)
	ee, = plt.plot(e_range, e, color='k', marker="P", label="$\sigma=0.5,\mu=0$", linewidth=0.5)
	# ff, = plt.plot(f_range, f, color='b', marker='.', label="$\sigma=0.4,\mu=0$", linewidth=0.5)
	gg, = plt.plot(g_range, g, color='r', marker='d', label="$\sigma=0.3,\mu=0$", linewidth=0.5)
	hh, = plt.plot(h_range, h, color='c', marker=".", label="$\sigma=0.2,\mu=0$", linewidth=0.5)
	jj, = plt.plot(j_range, j, color='m', marker=".", label="$\sigma=0.1,\mu=0$", linewidth=0.5)
	ii, = plt.plot(j_range, i, color='y', marker="^", label="$\sigma=0.15,\mu=0$", linewidth=0.5)
	kk, = plt.plot(k_range, k, color='g', marker="*", label="$\sigma=1.0,\mu=0$", linewidth=0.5)
	mean_line = plt.plot(e_range, np.ones_like(e_range) * 0.98, label='Benchmark', linestyle='--')
	
	# plt.legend(handler_map={aa: HandlerLine2D(numpoints=1)})
	# plt.legend([aa, bb, cc, dd], ["Multimodal Uniform ", "Multimodal Gaussian", "Uniform", "Gaussian"],
	#            handler_map={aa: HandlerLine2D(numpoints=1), bb: HandlerLine2D(numpoints=1), cc: HandlerLine2D(numpoints=1),
	#                         dd: HandlerLine2D(numpoints=1)
	
	# }, loc='middle right')
	plt.legend(loc='best')
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy Score")
	# plt.axis("auto")
	plt.grid(True)
	plt.show()
	plt.savefig("MMinfoGAN_mnist_GaussianSampler.png")
	plt.close()


def MM_zoom_plot_from_pkl():
	import numpy as np
	import matplotlib.pyplot as plt
	import pickle
	plt.Figure(figsize=(15, 15))
	
	plt.title('Zoom_MMinfoGAN_mnist_MultivariateGaussianSampler', fontsize=12)
	# a= pickle.load(open(dir+"classifier_MMinfoGAN_mnist_MultivariateMultivariateGaussianSampler_mu_0.0_sigma_10.0_accuracy.pkl", "rb"))[2:]
	# b= pickle.load(open(dir+"classifier_MMinfoGAN_mnist_MultivariateMultivariateGaussianSampler_mu_0.0_sigma_5.0_accuracy.pkl", "rb"))[2:]
	# c= pickle.load(open(dir+"classifier_MMinfoGAN_mnist_MultivariateMultivariateGaussianSampler_mu_0.0_sigma_3.0_accuracy.pkl", "rb"))[2:]
	# d= pickle.load(open(dir+"classifier_MMinfoGAN_mnist_MultivariateMultivariateGaussianSampler_mu_0.0_sigma_2.0_accuracy.pkl", "rb"))[2:]
	e = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_MultivariateGaussianSampler_mu_0.0_sigma_0.5_accuracy.pkl", "rb"))[2:]
	f = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_MultivariateGaussianSampler_mu_0.0_sigma_0.4_accuracy.pkl", "rb"))[2:]
	g = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_MultivariateGaussianSampler_mu_0.0_sigma_0.3_accuracy.pkl", "rb"))[2:]
	h = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_MultivariateGaussianSampler_mu_0.0_sigma_0.2_accuracy.pkl", "rb"))[2:]
	i = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_MultivariateGaussianSampler_mu_0.0_sigma_0.1_accuracy.pkl", "rb"))[2:]
	j = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_MultivariateGaussianSampler_mu_0.0_sigma_0.15_accuracy.pkl", "rb"))[2:]
	k = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_MultivariateGaussianSampler_mu_0.0_sigma_1.0_accuracy.pkl", "rb"))[2:]
	# plt.plot(a, np.arange(len(a)), 'r--',  b,np.arange(len(b)), 'b--',  c,np.arange(len(c)),'g^',d,np.arange(len(d)),"y--")
	# a_range = np.arange(2,len(a)+2)
	# b_range = np.arange(2,len(b)+2)
	# c_range = np.arange(2,len(c)+2)
	# d_range = np.arange(2,len(d)+2)
	e_range = np.arange(2, len(e) + 2)
	f_range = np.arange(2, len(f) + 2)
	g_range = np.arange(2, len(g) + 2)
	h_range = np.arange(2, len(h) + 2)
	i_range = np.arange(len(i))
	j_range = np.arange(len(j))
	k_range = np.arange(len(k))
	# aa, = plt.plot(a_range, a, color='b', marker="P", label="$\sigma=10,\mu=0$", linewidth=0.5)
	# bb, = plt.plot(b_range, b, color='g', marker='d', label="$\sigma=5,\mu=0$", linewidth=0.5)
	# # cc, = plt.plot(c_range, c, color='r', marker='^', label="$\sigma=3,\mu=0$", linewidth=0.5)
	# dd, = plt.plot(d_range, d, color='y', marker=".", label="$\sigma=2,\mu=0$", linewidth=0.5)
	ee, = plt.plot(e_range, e, color='k', marker="P", label="$\sigma=0.5,\mu=0$", linewidth=0.5)
	ff, = plt.plot(f_range, f, color='b', marker='.', label="$\sigma=0.4,\mu=0$", linewidth=0.5)
	gg, = plt.plot(g_range, g, color='r', marker='d', label="$\sigma=0.3,\mu=0$", linewidth=0.5)
	hh, = plt.plot(h_range, h, color='c', marker=".", label="$\sigma=0.2,\mu=0$", linewidth=0.5)
	# ii, = plt.plot(j_range, i, color='y', marker="^", label="$\sigma=0.1,\mu=0$", linewidth=0.5)
	# jj, = plt.plot(j_range, j, color='m', marker=".", label="$\sigma=0.15,\mu=0$", linewidth=0.5)
	kk, = plt.plot(k_range, k, color='g', marker="*", label="$\sigma=1.0,\mu=0$", linewidth=0.5)
	
	# plt.legend(handler_map={aa: HandlerLine2D(numpoints=1)})
	# plt.legend([aa, bb, cc, dd], ["Multimodal Uniform ", "Multimodal Gaussian", "Uniform", "Gaussian"],
	#            handler_map={aa: HandlerLine2D(numpoints=1), bb: HandlerLine2D(numpoints=1), cc: HandlerLine2D(numpoints=1),
	#                         dd: HandlerLine2D(numpoints=1)
	
	# }, loc='middle right')
	plt.legend(loc='best')
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy Score")
	# plt.axis("auto")
	plt.grid(True)
	plt.show()
	plt.savefig("Zoom_MMinfoGAN_mnist_MultivariateGaussianSampler.png")
	plt.close()


def MM_plot_from_pkl():
	import numpy as np
	import matplotlib.pyplot as plt
	import pickle
	plt.Figure(figsize=(15, 15))
	
	plt.title('MMinfoGAN_mnist_MultivariateGaussianSampler', fontsize=12)
	# a= pickle.load(open(dir+"classifier_MMinfoGAN_mnist_MultivariateMultivariateGaussianSampler_mu_0.0_sigma_10.0_accuracy.pkl", "rb"))
	# b= pickle.load(open(dir+"classifier_MMinfoGAN_mnist_MultivariateMultivariateGaussianSampler_mu_0.0_sigma_5.0_accuracy.pkl", "rb"))
	# c= pickle.load(open(dir+"classifier_MMinfoGAN_mnist_MultivariateMultivariateGaussianSampler_mu_0.0_sigma_3.0_accuracy.pkl", "rb"))
	# d= pickle.load(open(dir+"classifier_MMinfoGAN_mnist_MultivariateMultivariateGaussianSampler_mu_0.0_sigma_2.0_accuracy.pkl", "rb"))
	e = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_MultivariateGaussianSampler_mu_0.0_sigma_0.5_accuracy.pkl", "rb"))
	f = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_MultivariateGaussianSampler_mu_0.0_sigma_0.4_accuracy.pkl", "rb"))
	g = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_MultivariateGaussianSampler_mu_0.0_sigma_0.3_accuracy.pkl", "rb"))
	h = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_MultivariateGaussianSampler_mu_0.0_sigma_0.2_accuracy.pkl", "rb"))
	i = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_MultivariateGaussianSampler_mu_0.0_sigma_0.1_accuracy.pkl", "rb"))
	j = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_MultivariateGaussianSampler_mu_0.0_sigma_0.15_accuracy.pkl", "rb"))
	k = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_MultivariateGaussianSampler_mu_0.0_sigma_1.0_accuracy.pkl", "rb"))
	# a_range = np.arange(len(a))
	# b_range = np.arange(len(b))
	# c_range = np.arange(len(c))
	# d_range = np.arange(len(d))
	e_range = np.arange(len(e))
	f_range = np.arange(len(f))
	g_range = np.arange(len(g))
	h_range = np.arange(len(h))
	i_range = np.arange(len(i))
	j_range = np.arange(len(j))
	k_range = np.arange(len(k))
	# aa, = plt.plot(a_range, a, color='b', marker="P", label="$\sigma=10,\mu=0$", linewidth=0.5)
	# bb, = plt.plot(b_range, b, color='g', marker='d', label="$\sigma=5,\mu=0$", linewidth=0.5)
	# cc, = plt.plot(c_range, c, color='r', marker='^', label="$\sigma=3,\mu=0$", linewidth=0.5)
	# dd, = plt.plot(d_range, d, color='y', marker=".", label="$\sigma=2,\mu=0$", linewidth=0.5)
	ee, = plt.plot(e_range, e, color='k', marker="P", label="$\sigma=0.5,\mu=0$", linewidth=0.5)
	ff, = plt.plot(f_range, f, color='b', marker='.', label="$\sigma=0.4,\mu=0$", linewidth=0.5)
	gg, = plt.plot(g_range, g, color='r', marker='d', label="$\sigma=0.3,\mu=0$", linewidth=0.5)
	hh, = plt.plot(h_range, h, color='c', marker=".", label="$\sigma=0.2,\mu=0$", linewidth=0.5)
	# jj, = plt.plot(j_range, j, color='m', marker=".", label="$\sigma=0.1,\mu=0$", linewidth=0.5)
	ii, = plt.plot(j_range, i, color='y', marker="^", label="$\sigma=0.15,\mu=0$", linewidth=0.5)
	kk, = plt.plot(k_range, k, color='g', marker="*", label="$\sigma=1.0,\mu=0$", linewidth=0.5)
	mean_line = plt.plot(e_range, np.ones_like(e_range) * 0.98, label='Benchmark', linestyle='--')
	
	# plt.legend(handler_map={aa: HandlerLine2D(numpoints=1)})
	# plt.legend([aa, bb, cc, dd], ["Multimodal Uniform ", "Multimodal Gaussian", "Uniform", "Gaussian"],
	#            handler_map={aa: HandlerLine2D(numpoints=1), bb: HandlerLine2D(numpoints=1), cc: HandlerLine2D(numpoints=1),
	#                         dd: HandlerLine2D(numpoints=1)
	
	# }, loc='middle right')
	plt.legend(loc='best')
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy Score")
	# plt.axis("auto")
	plt.grid(True)
	plt.show()
	plt.savefig("MMinfoGAN_mnist_MultivariateGaussianSampler.png")
	plt.close()


def MM_mu_05_07_08_zoom_plot_from_pkl():
	import numpy as np
	import matplotlib.pyplot as plt
	import pickle
	plt.Figure(figsize=(15, 15))
	
	title = '1_MMinfoGAN_Fsion-Mnist_multi-modal Gaussian Sampler 10 modals'
	plt.title(title, fontsize=10)
	START = 2
	a = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.5_sigma_0.17_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:50]
	b = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:50]
	c = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:50]
	d = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:50]
	e = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:50]
	f = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:50]
	g = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:50]
	# DIR 2
	a2 = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.5_sigma_0.17_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:50]
	b2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:50]
	c2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:50]
	d2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:50]
	e2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:50]
	f2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:50]
	g2 = pickle.load(open(dir2 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:50]
	
	# DIR 3
	# a3 = pickle.load(
	# 	open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.5_sigma_0.17_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:50]
	# b3 = pickle.load(
	# 	open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:50]
	c3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:50]
	d3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.7_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:50]
	# e3 = pickle.load(
	# 	open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:50]
	f3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:50]
	g3 = pickle.load(open(dir3 + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[START:50]
	a_range = np.arange(START, len(a) + START)
	b_range = np.arange(START, len(b) + START)
	c_range = np.arange(START, len(c) + START)
	d_range = np.arange(START, len(d) + START)
	# e_range = np.arange(START, len(e) + START)
	f_range = np.arange(START, len(f) + START)
	g_range = np.arange(START, len(g) + START)
	# h_range = np.arange(2, len(h) + 2)
	# i_range = np.arange(len(i))
	# j_range = np.arange(len(j))
	# k_range = np.arange(len(k))
	
	# CALC MEAN AND STDERR
	# a_mean = np.mean([a, a2, a3], axis=0)
	# a_stderr = np.std([a, a2, a3], axis=0) / np.sqrt(len(a))
	# b_mean = np.mean([b, b2, b3], axis=0)
	# b_stderr = np.std([b, b2, b3], axis=0) / np.sqrt(len(a))
	c_mean = np.mean([c, c2, c3], axis=0)
	c_stderr = np.std([c, c2, c3], axis=0) / np.sqrt(len(c))
	d_mean = np.mean([d, d2, d3], axis=0)
	d_stderr = np.std([d, d2, d3], axis=0) / np.sqrt(len(d))
	# e_mean = np.mean([e, e2, e3], axis=0)
	# e_stderr = np.std([e, e2, e3], axis=0) / np.sqrt(len(e))
	f_mean = np.mean([f, f2, f3], axis=0)
	f_stderr = np.std([f, f2, f3], axis=0) / np.sqrt(len(f))
	g_mean = np.mean([g, g2, g3], axis=0)
	g_stderr = np.std([g, g2, g3], axis=0) / np.sqrt(len(g))
	# plt.errorbar(a_range, a_mean, yerr=a_stderr, color='red', ls='--', marker='o', capsize=5, capthick=1, ecolor='black')
	# plt.errorbar(a_range, b_mean, yerr=b_stderr, color='red', ls='--', marker='o', capsize=5, capthick=1, ecolor='black')
	plt.errorbar(c_range, c_mean, yerr=c_stderr, color='g', marker='d', label="$\Sigma=0.15,\mu=0.7$", ls='--', capsize=5, capthick=1, ecolor='k')
	plt.errorbar(d_range, d_mean, yerr=d_stderr, color='c', marker='^', label="$\Sigma=0.25,\mu=0.7$", ls='--', capsize=5, capthick=1, ecolor='k')
	# plt.errorbar(a_range, e_mean, yerr=e_stderr, color='red', ls='--', marker='o', capsize=5, capthick=1, ecolor='black')
	# plt.errorbar(f_range, f_mean, yerr=f_stderr, color='b', marker='+', label="$\Sigma=0.2,\mu=0.8$", ls='--', capsize=5, capthick=1, ecolor='k')
	# plt.errorbar(g_range, g_mean, yerr=g_stderr, color='r', marker='d', label="$\Sigma=0.3,\mu=08$", ls='--', capsize=5, capthick=1, ecolor='k')
	
	# aa, = plt.plot(a_range, a, color='b', marker="P", label="$\Sigma=0.17,\mu=0.5$", linewidth=0.5)
	# bb, = plt.plot(b_range, b, color='g', marker='d', label="$\Sigma=0.15,\mu=0.7$", linewidth=0.5)
	cc, = plt.plot(c_range, c, color='r', marker='^', label="$\Sigma=0.25,\mu=0.7$", linewidth=0.5)
	dd, = plt.plot(d_range, d, color='r', marker=".", label="$\Sigma=0.3,\mu=0.7$", linewidth=0.5)
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
	
	title = '2_MMinfoGAN_Fsion-Mnist_multi-modal Gaussian Sampler 10 modals'
	plt.title(title, fontsize=10)
	a = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.1_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[2:50]
	b = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.13_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[2:50]
	c = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[2:50]
	d = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.17_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[2:50]
	e = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[2:50]
	f = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[2:50]
	# g = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[2:50]
	# h = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.1_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[2:50]
	# i = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.1_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[2:50]
	# j = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.1_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[2:50]
	# k = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.1_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[2:50]
	# plt.plot(a, np.arange(len(a)), 'r--', b, np.arange(len(b)), 'b--', c, np.arange(len(c)), 'g^', d, np.arange(len(d)), "y--")
	a_range = np.arange(2, len(a) + 2)
	b_range = np.arange(2, len(b) + 2)
	c_range = np.arange(2, len(c) + 2)
	d_range = np.arange(2, len(d) + 2)
	e_range = np.arange(2, len(e) + 2)
	f_range = np.arange(2, len(f) + 2)
	# g_range = np.arange(2, len(g) + 2)
	# h_range = np.arange(2, len(h) + 2)
	# i_range = np.arange(len(i))
	# j_range = np.arange(len(j))
	# k_range = np.arange(len(k))
	aa, = plt.plot(a_range, a, color='b', marker="P", label="$\Sigma=0.1,\mu=0.1$", linewidth=0.5)
	bb, = plt.plot(b_range, b, color='g', marker='d', label="$\Sigma=0.13,\mu=0.1$", linewidth=0.5)
	cc, = plt.plot(c_range, c, color='r', marker='^', label="$\Sigma=0.15,\mu=0.1$", linewidth=0.5)
	dd, = plt.plot(d_range, d, color='y', marker="+", label="$\Sigma=0.17,\mu=0.1$", linewidth=0.5)
	ee, = plt.plot(e_range, e, color='k', marker="P", label="$\Sigma=0.2,\mu=0.1$", linewidth=0.5)
	ff, = plt.plot(f_range, f, color='c', marker='.', label="$\Sigma=0.25,\mu=0.1$", linewidth=0.5)
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
	
	title = '3_MMinfoGAN_Fsion-Mnist_multi-modal Gaussian Sampler 10 modals'
	plt.title(title, fontsize=10)
	a = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.15_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[2:50]
	b = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.2_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[2:50]
	# c = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.22_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[2:50]
	# d = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.25_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[2:50]
	e = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.3_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[2:50]
	f = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.4_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[2:50]
	g = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.5_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[2:50]
	# h = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.5_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[2:50]
	# i = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.1_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[2:50]
	# j = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.1_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[2:50]
	# k = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.1_czcc_czrc_rzcc_rzrc_ndist_10_accuracy.pkl", "rb"))[2:50]
	# plt.plot(a, np.arange(len(a)), 'r--', b, np.arange(len(b)), 'b--', c, np.arange(len(c)), 'g^', d, np.arange(len(d)), "y--")
	a_range = np.arange(2, len(a) + 2)
	b_range = np.arange(2, len(b) + 2)
	# c_range = np.arange(2, len(c) + 2)
	# d_range = np.arange(2, len(d) + 2)
	e_range = np.arange(2, len(e) + 2)
	f_range = np.arange(2, len(f) + 2)
	g_range = np.arange(2, len(g) + 2)
	# h_range = np.arange(2, len(h) + 2)
	# i_range = np.arange(len(i))
	# j_range = np.arange(len(j))
	# k_range = np.arange(len(k))
	aa, = plt.plot(a_range, a, color='b', marker="P", label="$\Sigma=0.15,\mu=1.0$", linewidth=0.5)
	bb, = plt.plot(b_range, b, color='g', marker='d', label="$\Sigma=0.2,\mu=1.0$", linewidth=0.5)
	# cc, = plt.plot(c_range, c, color='r', marker='^', label="$\sigma=0.22,\mu=1.0$", linewidth=0.5)
	# dd, = plt.plot(d_range, d, color='y', marker=".", label="$\sigma=0.25,\mu=1.0$", linewidth=0.5)
	ee, = plt.plot(e_range, e, color='k', marker="P", label="$\Sigma=0.3,\mu=1.0$", linewidth=0.5)
	ff, = plt.plot(f_range, f, color='y', marker='^', label="$\Sigma=0.4,\mu=1.0$", linewidth=0.5)
	gg, = plt.plot(g_range, g, color='r', marker='d', label="$\Sigma=0.5,\mu=1.0$", linewidth=0.5)
	# hh, = plt.plot(h_range, h, color='c', marker=".", label="$\sigma=0.2,\mu=1.0$", linewidth=0.5)
	# ii, = plt.plot(i_range, i, color='y', marker="^", label="$\sigma=0.1,\mu=0$", linewidth=0.5)
	# jj, = plt.plot(j_range, j, color='m', marker=".", label="$\sigma=0.15,\mu=0$", linewidth=0.5)
	# kk, = plt.plot(k_range, k, color='g', marker="*", label="$\sigma=1.0,\mu=0$", linewidth=0.5)
	
	# plt.legend(handler_map={aa: HandlerLine2D(numpoints=1)})
	# plt.legend([aa, bb, cc, dd], ["Multimodal Uniform ", "Multimodal Gaussian", "Uniform", "Gaussian"],
	#            handler_map={aa: HandlerLine2D(numpoints=1), bb: HandlerLine2D(numpoints=1), cc: HandlerLine2D(numpoints=1),
	#                         dd: HandlerLine2D(numpoints=1)
	
	# }, loc='middle right')
	plt.legend(loc='best')
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy Score")
	# plt.axis("auto")
	plt.grid(True)
	plt.show()
	plt.savefig(title + ".png")
	plt.close()


def MM_mu_1plot_from_pkl():
	import numpy as np
	import matplotlib.pyplot as plt
	import pickle
	plt.Figure(figsize=(15, 15))
	
	plt.title('MMinfoGAN_mnist_MultivariateGaussianSampler $\mu=0.1$', fontsize=12)
	# a= pickle.load(open(dir+"classifier_MMinfoGAN_mnist_MultivariateMultivariateGaussianSampler_mu_0.1_sigma_10.0_accuracy.pkl", "rb"))
	# b= pickle.load(open(dir+"classifier_MMinfoGAN_mnist_MultivariateMultivariateGaussianSampler_mu_0.1_sigma_5.0_accuracy.pkl", "rb"))
	# c= pickle.load(open(dir+"classifier_MMinfoGAN_mnist_MultivariateMultivariateGaussianSampler_mu_0.1_sigma_3.0_accuracy.pkl", "rb"))
	# d= pickle.load(open(dir+"classifier_MMinfoGAN_mnist_MultivariateMultivariateGaussianSampler_mu_0.1_sigma_2.0_accuracy.pkl", "rb"))
	e = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.5_accuracy.pkl", "rb"))
	# f = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.4_accuracy.pkl", "rb"))
	# g = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.3_accuracy.pkl", "rb"))
	h = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.2_accuracy.pkl", "rb"))
	# i = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.1_accuracy.pkl", "rb"))
	j = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.15_accuracy.pkl", "rb"))
	k = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_MultivariateGaussianSampler_mu_0.1_sigma_1.0_accuracy.pkl", "rb"))
	# a_range = np.arange(len(a))
	# b_range = np.arange(len(b))
	# c_range = np.arange(len(c))
	# d_range = np.arange(len(d))
	e_range = np.arange(len(e))
	# f_range = np.arange(len(f))
	# g_range = np.arange(len(g))
	h_range = np.arange(len(h))
	# i_range = np.arange(len(i))
	j_range = np.arange(len(j))
	k_range = np.arange(len(k))
	# aa, = plt.plot(a_range, a, color='b', marker="P", label="$\sigma=10,\mu=0$", linewidth=0.5)
	# bb, = plt.plot(b_range, b, color='g', marker='d', label="$\sigma=5,\mu=0$", linewidth=0.5)
	# cc, = plt.plot(c_range, c, color='r', marker='^', label="$\sigma=3,\mu=0$", linewidth=0.5)
	# dd, = plt.plot(d_range, d, color='y', marker=".", label="$\sigma=2,\mu=0$", linewidth=0.5)
	ee, = plt.plot(e_range, e, color='k', marker="P", label="$\sigma=0.5,\mu=0$", linewidth=0.5)
	# ff, = plt.plot(f_range, f, color='b', marker='.', label="$\sigma=0.4,\mu=0$", linewidth=0.5)
	# gg, = plt.plot(g_range, g, color='r', marker='d', label="$\sigma=0.3,\mu=0$", linewidth=0.5)
	hh, = plt.plot(h_range, h, color='c', marker=".", label="$\sigma=0.2,\mu=0$", linewidth=0.5)
	# jj, = plt.plot(j_range, j, color='m', marker=".", label="$\sigma=0.1,\mu=0$", linewidth=0.5)
	# ii, = plt.plot(j_range, i, color='y', marker="^", label="$\sigma=0.15,\mu=0$", linewidth=0.5)
	kk, = plt.plot(k_range, k, color='g', marker="*", label="$\sigma=1.0,\mu=0$", linewidth=0.5)
	mean_line = plt.plot(e_range, np.ones_like(e_range) * 0.98, label='Benchmark', linestyle='--')
	
	# plt.legend(handler_map={aa: HandlerLine2D(numpoints=1)})
	# plt.legend([aa, bb, cc, dd], ["Multimodal Uniform ", "Multimodal Gaussian", "Uniform", "Gaussian"],
	#            handler_map={aa: HandlerLine2D(numpoints=1), bb: HandlerLine2D(numpoints=1), cc: HandlerLine2D(numpoints=1),
	#                         dd: HandlerLine2D(numpoints=1)
	
	# }, loc='middle right')
	plt.legend(loc='best')
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy Score")
	# plt.axis("auto")
	plt.grid(True)
	plt.show()
	plt.savefig("MMinfoGAN_mnist_MultivariateGaussianSampler $\mu=0.1$.png")
	plt.close()


# ACCURACY fashion
def truncated_fashion__zoom_plot_from_pkl():
	import numpy as np
	import matplotlib.pyplot as plt
	import pickle
	plt.Figure(figsize=(15, 15))
	
	plt.title('Zoom_MMinfoGAN_fashion-mnist_TruncatedGaussianSample', fontsize=12)
	a = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_TruncatedGaussianSample_mu_0.0_sigma_10.0_accuracy.pkl", "rb"))[2:]
	b = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_TruncatedGaussianSample_mu_0.0_sigma_5.0_accuracy.pkl", "rb"))[2:]
	c = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_TruncatedGaussianSample_mu_0.0_sigma_3.0_accuracy.pkl", "rb"))[2:]
	d = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_TruncatedGaussianSample_mu_0.0_sigma_2.0_accuracy.pkl", "rb"))[2:50]
	# e = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_TruncatedGaussianSample_mu_0.0_sigma_0.5_accuracy.pkl", "rb"))[2:]
	# f = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_TruncatedGaussianSample_mu_0.0_sigma_0.4_accuracy.pkl", "rb"))[2:]
	# g = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_TruncatedGaussianSample_mu_0.0_sigma_0.3_accuracy.pkl", "rb"))[2:]
	# h = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_TruncatedGaussianSample_mu_0.0_sigma_0.2_accuracy.pkl", "rb"))[2:]
	# i= pickle.load(open(dir+"classifier_MMinfoGAN_fashion-mnist_TruncatedGaussianSample_mu_0.0_sigma_0.1_accuracy.pkl", "rb"))
	# j = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_TruncatedGaussianSample_mu_0.0_sigma_0.15_accuracy.pkl", "rb"))[2:]
	# k = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_TruncatedGaussianSample_mu_0.0_sigma_1.0_accuracy.pkl", "rb"))[2:]
	# plt.plot(a, np.arange(len(a)), 'r--',  b,np.arange(len(b)), 'b--',  c,np.arange(len(c)),'g^',d,np.arange(len(d)),"y--")
	a_range = np.arange(2, len(a) + 2)
	b_range = np.arange(2, len(b) + 2)
	c_range = np.arange(2, len(c) + 2)
	d_range = np.arange(2, len(d) + 2)
	# e_range = np.arange(2, len(e) + 2)
	# f_range = np.arange(2, len(f) + 2)
	# g_range = np.arange(2, len(g) + 2)
	# h_range = np.arange(2, len(h) + 2)
	# i_range = np.arange(len(i))
	# j_range = np.arange(len(j))
	# k_range = np.arange(len(k))
	aa, = plt.plot(a_range, a, color='b', marker="P", label="$\sigma=10,\mu=0$", linewidth=0.5)
	bb, = plt.plot(b_range, b, color='g', marker='d', label="$\sigma=5,\mu=0$", linewidth=0.5)
	# cc, = plt.plot(c_range, c, color='r', marker='^', label="$\sigma=3,\mu=0$", linewidth=0.5)
	dd, = plt.plot(d_range, d, color='y', marker=".", label="$\sigma=2,\mu=0$", linewidth=0.5)
	# ee, = plt.plot(e_range, e, color='k', marker="P", label="$\sigma=0.5,\mu=0$", linewidth=0.5)
	# ff, = plt.plot(f_range, f, color='g', marker='.', label="$\sigma=0.4,\mu=0$", linewidth=0.5)
	# gg, = plt.plot(g_range, g, color='r', marker='>', label="$\sigma=0.3,\mu=0$", linewidth=0.5)
	# hh, = plt.plot(h_range, h, color='c', marker=".", label="$\sigma=0.2,\mu=0$", linewidth=0.5)
	# jj, = plt.plot(j_range, j, color='m', marker=".", label="$\sigma=0.15,\mu=0$", linewidth=0.5)
	# kk, = plt.plot(k_range, k, color='k', marker="*", label="$\sigma=1.0,\mu=0$", linewidth=0.5)
	mean_line = plt.plot(c_range, np.ones_like(d_range) * 0.98, label='Benchmark', linestyle='--')
	
	# plt.legend(handler_map={aa: HandlerLine2D(numpoints=1)})
	# plt.legend([aa, bb, cc, dd], ["Multimodal Uniform ", "Multimodal Gaussian", "Uniform", "Gaussian"],
	#            handler_map={aa: HandlerLine2D(numpoints=1), bb: HandlerLine2D(numpoints=1), cc: HandlerLine2D(numpoints=1),
	#                         dd: HandlerLine2D(numpoints=1)
	
	# }, loc='middle right')
	plt.legend(loc='best')
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy Score")
	# plt.axis("auto")
	plt.grid(True)
	plt.show()
	plt.savefig("Zoom_MMinfoGAN_fashion-mnist_TruncatedGaussianSample.png")
	plt.close()


def fashion_truncated_plot_from_pkl():
	import numpy as np
	import matplotlib.pyplot as plt
	import pickle
	plt.Figure(figsize=(15, 15))
	
	plt.title('MMinfoGAN_fashion-mnist_TruncatedGaussianSample', fontsize=12)
	a = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_TruncatedGaussianSample_mu_0.0_sigma_10.0_accuracy.pkl", "rb"))
	b = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_TruncatedGaussianSample_mu_0.0_sigma_5.0_accuracy.pkl", "rb"))
	c = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_TruncatedGaussianSample_mu_0.0_sigma_3.0_accuracy.pkl", "rb"))
	d = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_TruncatedGaussianSample_mu_0.0_sigma_2.0_accuracy.pkl", "rb"))
	# e = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_TruncatedGaussianSample_mu_0.0_sigma_0.5_accuracy.pkl", "rb"))
	# f = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_TruncatedGaussianSample_mu_0.0_sigma_0.4_accuracy.pkl", "rb"))
	# g = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_TruncatedGaussianSample_mu_0.0_sigma_0.3_accuracy.pkl", "rb"))
	# h = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_TruncatedGaussianSample_mu_0.0_sigma_0.2_accuracy.pkl", "rb"))
	# i= pickle.load(open(dir+"classifier_MMinfoGAN_fashion-mnist_TruncatedGaussianSample_mu_0.0_sigma_0.1_accuracy.pkl", "rb"))
	# j = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_TruncatedGaussianSample_mu_0.0_sigma_0.15_accuracy.pkl", "rb"))
	# k = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_TruncatedGaussianSample_mu_0.0_sigma_1.0_accuracy.pkl", "rb"))
	# plt.plot(a, np.arange(len(a)), 'r--',  b,np.arange(len(b)), 'b--',  c,np.arange(len(c)),'g^',d,np.arange(len(d)),"y--")
	a_range = np.arange(len(a))
	b_range = np.arange(len(b))
	c_range = np.arange(len(c))
	d_range = np.arange(len(d))
	# e_range = np.arange(len(e))
	# f_range = np.arange(len(f))
	# g_range = np.arange(len(g))
	# h_range = np.arange(len(h))
	# i_range = np.arange(len(i))
	# j_range = np.arange(len(j))
	# k_range = np.arange(len(k))
	aa, = plt.plot(a_range, a, color='b', marker="P", label="$\sigma=10,\mu=0$", linewidth=0.5)
	# bb, = plt.plot(b_range, b, color='g', marker='d', label="$\sigma=5,\mu=0$", linewidth=0.5)
	# cc, = plt.plot(c_range, c, color='r', marker='^', label="$\sigma=3,\mu=0$", linewidth=0.5)
	dd, = plt.plot(d_range, d, color='y', marker=".", label="$\sigma=2,\mu=0$", linewidth=0.5)
	# ee, = plt.plot(e_range, e, color='k', marker="P", label="$\sigma=0.5,\mu=0$", linewidth=0.5)
	# ff, = plt.plot(f_range, f, color='g', marker='.', label="$\sigma=0.4,\mu=0$", linewidth=0.5)
	# gg, = plt.plot(g_range, g, color='c', marker='>', label="$\sigma=0.3,\mu=0$", linewidth=0.5)
	# hh, = plt.plot(h_range, h, color='c', marker=".", label="$\sigma=0.2,\mu=0$", linewidth=0.5)
	# jj, = plt.plot(j_range, j, color='m', marker=".", label="$\sigma=0.15,\mu=0$", linewidth=0.5)
	# kk, = plt.plot(k_range, k, color='g', marker="*", label="$\sigma=1.0,\mu=0$", linewidth=0.5)
	mean_line = plt.plot(c_range, np.ones_like(d_range) * 0.98, label='Benchmark', linestyle='--')
	
	# plt.legend(handler_map={aa: HandlerLine2D(numpoints=1)})
	# plt.legend([aa, bb, cc, dd], ["Multimodal Uniform ", "Multimodal Gaussian", "Uniform", "Gaussian"],
	#            handler_map={aa: HandlerLine2D(numpoints=1), bb: HandlerLine2D(numpoints=1), cc: HandlerLine2D(numpoints=1),
	#                         dd: HandlerLine2D(numpoints=1)
	
	# }, loc='middle right')
	plt.legend(loc='best')
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy Score")
	# plt.axis("auto")
	plt.grid(True)
	plt.show()
	plt.savefig("MMinfoGAN_fashion-mnist_TruncatedGaussianSample.png")
	plt.close()


def fashion_gaussian_zoom_plot_from_pkl():
	import numpy as np
	import matplotlib.pyplot as plt
	import pickle
	plt.Figure(figsize=(15, 15))
	
	plt.title('Zoom_MMinfoGAN_fashion-mnist_GaussianSample', fontsize=12)
	# a= pickle.load(open(dir+"classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSample_mu_0.0_sigma_10.0_accuracy.pkl", "rb"))[2:]
	# b= pickle.load(open(dir+"classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSample_mu_0.0_sigma_5.0_accuracy.pkl", "rb"))[2:]
	# c= pickle.load(open(dir+"classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSample_mu_0.0_sigma_3.0_accuracy.pkl", "rb"))[2:]
	# d= pickle.load(open(dir+"classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSample_mu_0.0_sigma_2.0_accuracy.pkl", "rb"))[2:]
	e = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_GaussianSample_mu_0.0_sigma_0.5_accuracy.pkl", "rb"))[2:]
	f = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_GaussianSample_mu_0.0_sigma_0.4_accuracy.pkl", "rb"))[2:]
	g = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_GaussianSample_mu_0.0_sigma_0.3_accuracy.pkl", "rb"))[2:]
	h = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_GaussianSample_mu_0.0_sigma_0.2_accuracy.pkl", "rb"))[2:]
	i = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_GaussianSample_mu_0.0_sigma_0.1_accuracy.pkl", "rb"))[2:]
	j = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_GaussianSample_mu_0.0_sigma_0.15_accuracy.pkl", "rb"))[2:]
	k = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_GaussianSample_mu_0.0_sigma_1.0_accuracy.pkl", "rb"))[2:]
	# plt.plot(a, np.arange(len(a)), 'r--',  b,np.arange(len(b)), 'b--',  c,np.arange(len(c)),'g^',d,np.arange(len(d)),"y--")
	# a_range = np.arange(2,len(a)+2)
	# b_range = np.arange(2,len(b)+2)
	# c_range = np.arange(2,len(c)+2)
	# d_range = np.arange(2,len(d)+2)
	e_range = np.arange(2, len(e) + 2)
	f_range = np.arange(2, len(f) + 2)
	g_range = np.arange(2, len(g) + 2)
	h_range = np.arange(2, len(h) + 2)
	i_range = np.arange(len(i))
	j_range = np.arange(len(j))
	k_range = np.arange(len(k))
	# aa, = plt.plot(a_range, a, color='b', marker="P", label="$\sigma=10,\mu=0$", linewidth=0.5)
	# bb, = plt.plot(b_range, b, color='g', marker='d', label="$\sigma=5,\mu=0$", linewidth=0.5)
	# # cc, = plt.plot(c_range, c, color='r', marker='^', label="$\sigma=3,\mu=0$", linewidth=0.5)
	# dd, = plt.plot(d_range, d, color='y', marker=".", label="$\sigma=2,\mu=0$", linewidth=0.5)
	ee, = plt.plot(e_range, e, color='k', marker="P", label="$\sigma=0.5,\mu=0$", linewidth=0.5)
	ff, = plt.plot(f_range, f, color='b', marker='.', label="$\sigma=0.4,\mu=0$", linewidth=0.5)
	# gg, = plt.plot(g_range, g, color='r', marker='d', label="$\sigma=0.3,\mu=0$", linewidth=0.5)
	hh, = plt.plot(h_range, h, color='c', marker=".", label="$\sigma=0.2,\mu=0$", linewidth=0.5)
	ii, = plt.plot(j_range, i, color='y', marker="^", label="$\sigma=0.1,\mu=0$", linewidth=0.5)
	jj, = plt.plot(j_range, j, color='m', marker=".", label="$\sigma=0.15,\mu=0$", linewidth=0.5)
	kk, = plt.plot(k_range, k, color='g', marker="*", label="$\sigma=1.0,\mu=0$", linewidth=0.5)
	
	# plt.legend(handler_map={aa: HandlerLine2D(numpoints=1)})
	# plt.legend([aa, bb, cc, dd], ["Multimodal Uniform ", "Multimodal Gaussian", "Uniform", "Gaussian"],
	#            handler_map={aa: HandlerLine2D(numpoints=1), bb: HandlerLine2D(numpoints=1), cc: HandlerLine2D(numpoints=1),
	#                         dd: HandlerLine2D(numpoints=1)
	
	# }, loc='middle right')
	plt.legend(loc='best')
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy Score")
	# plt.axis("auto")
	plt.grid(True)
	plt.show()
	plt.savefig("Zoom_MMinfoGAN_fashion-mnist_GaussianSample.png")
	plt.close()


def fashion_gaussian_plot_from_pkl():
	import numpy as np
	import matplotlib.pyplot as plt
	import pickle
	plt.Figure(figsize=(15, 15))
	
	plt.title('MMinfoGAN_fashion-mnist_GaussianSample', fontsize=12)
	# a= pickle.load(open(dir+"classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSample_mu_0.0_sigma_10.0_accuracy.pkl", "rb"))
	# b= pickle.load(open(dir+"classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSample_mu_0.0_sigma_5.0_accuracy.pkl", "rb"))
	# c= pickle.load(open(dir+"classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSample_mu_0.0_sigma_3.0_accuracy.pkl", "rb"))
	# d= pickle.load(open(dir+"classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSample_mu_0.0_sigma_2.0_accuracy.pkl", "rb"))
	e = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_GaussianSample_mu_0.0_sigma_0.5_accuracy.pkl", "rb"))
	f = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_GaussianSample_mu_0.0_sigma_0.4_accuracy.pkl", "rb"))
	g = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_GaussianSample_mu_0.0_sigma_0.3_accuracy.pkl", "rb"))
	h = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_GaussianSample_mu_0.0_sigma_0.2_accuracy.pkl", "rb"))
	i = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_GaussianSample_mu_0.0_sigma_0.1_accuracy.pkl", "rb"))
	j = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_GaussianSample_mu_0.0_sigma_0.15_accuracy.pkl", "rb"))
	k = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_GaussianSample_mu_0.0_sigma_1.0_accuracy.pkl", "rb"))
	# a_range = np.arange(len(a))
	# b_range = np.arange(len(b))
	# c_range = np.arange(len(c))
	# d_range = np.arange(len(d))
	e_range = np.arange(len(e))
	f_range = np.arange(len(f))
	g_range = np.arange(len(g))
	h_range = np.arange(len(h))
	i_range = np.arange(len(i))
	j_range = np.arange(len(j))
	k_range = np.arange(len(k))
	# aa, = plt.plot(a_range, a, color='b', marker="P", label="$\sigma=10,\mu=0$", linewidth=0.5)
	# bb, = plt.plot(b_range, b, color='g', marker='d', label="$\sigma=5,\mu=0$", linewidth=0.5)
	# cc, = plt.plot(c_range, c, color='r', marker='^', label="$\sigma=3,\mu=0$", linewidth=0.5)
	# dd, = plt.plot(d_range, d, color='y', marker=".", label="$\sigma=2,\mu=0$", linewidth=0.5)
	ee, = plt.plot(e_range, e, color='k', marker="P", label="$\sigma=0.5,\mu=0$", linewidth=0.5)
	# ff, = plt.plot(f_range, f, color='b', marker='.', label="$\sigma=0.4,\mu=0$", linewidth=0.5)
	gg, = plt.plot(g_range, g, color='r', marker='d', label="$\sigma=0.3,\mu=0$", linewidth=0.5)
	hh, = plt.plot(h_range, h, color='c', marker=".", label="$\sigma=0.2,\mu=0$", linewidth=0.5)
	jj, = plt.plot(j_range, j, color='m', marker=".", label="$\sigma=0.1,\mu=0$", linewidth=0.5)
	ii, = plt.plot(j_range, i, color='y', marker="^", label="$\sigma=0.15,\mu=0$", linewidth=0.5)
	kk, = plt.plot(k_range, k, color='g', marker="*", label="$\sigma=1.0,\mu=0$", linewidth=0.5)
	mean_line = plt.plot(e_range, np.ones_like(e_range) * 0.98, label='Benchmark', linestyle='--')
	
	# plt.legend(handler_map={aa: HandlerLine2D(numpoints=1)})
	# plt.legend([aa, bb, cc, dd], ["Multimodal Uniform ", "Multimodal Gaussian", "Uniform", "Gaussian"],
	#            handler_map={aa: HandlerLine2D(numpoints=1), bb: HandlerLine2D(numpoints=1), cc: HandlerLine2D(numpoints=1),
	#                         dd: HandlerLine2D(numpoints=1)
	
	# }, loc='middle right')
	plt.legend(loc='best')
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy Score")
	# plt.axis("auto")
	plt.grid(True)
	plt.show()
	plt.savefig("MMinfoGAN_fashion-mnist_GaussianSampler.png")
	plt.close()


def fashion_MM_zoom_plot_from_pkl():
	import numpy as np
	import matplotlib.pyplot as plt
	import pickle
	plt.Figure(figsize=(15, 15))
	
	plt.title('Zoom_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler', fontsize=12)
	# a= pickle.load(open(dir+"classifier_MMinfoGAN_fashion-mnist_MultivariateMultivariateGaussianSampler_mu_0.0_sigma_10.0_accuracy.pkl", "rb"))[2:]
	# b= pickle.load(open(dir+"classifier_MMinfoGAN_fashion-mnist_MultivariateMultivariateGaussianSampler_mu_0.0_sigma_5.0_accuracy.pkl", "rb"))[2:]
	# c= pickle.load(open(dir+"classifier_MMinfoGAN_fashion-mnist_MultivariateMultivariateGaussianSampler_mu_0.0_sigma_3.0_accuracy.pkl", "rb"))[2:]
	# d= pickle.load(open(dir+"classifier_MMinfoGAN_fashion-mnist_MultivariateMultivariateGaussianSampler_mu_0.0_sigma_2.0_accuracy.pkl", "rb"))[2:]
	e = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.0_sigma_0.5_accuracy.pkl", "rb"))[2:]
	f = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.0_sigma_0.4_accuracy.pkl", "rb"))[2:]
	g = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.0_sigma_0.3_accuracy.pkl", "rb"))[2:]
	h = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.0_sigma_0.2_accuracy.pkl", "rb"))[2:]
	i = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.0_sigma_0.1_accuracy.pkl", "rb"))[2:]
	j = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.0_sigma_0.15_accuracy.pkl", "rb"))[2:]
	k = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.0_sigma_1.0_accuracy.pkl", "rb"))[2:]
	# plt.plot(a, np.arange(len(a)), 'r--',  b,np.arange(len(b)), 'b--',  c,np.arange(len(c)),'g^',d,np.arange(len(d)),"y--")
	# a_range = np.arange(2,len(a)+2)
	# b_range = np.arange(2,len(b)+2)
	# c_range = np.arange(2,len(c)+2)
	# d_range = np.arange(2,len(d)+2)
	e_range = np.arange(2, len(e) + 2)
	f_range = np.arange(2, len(f) + 2)
	g_range = np.arange(2, len(g) + 2)
	h_range = np.arange(2, len(h) + 2)
	i_range = np.arange(len(i))
	j_range = np.arange(len(j))
	k_range = np.arange(len(k))
	# aa, = plt.plot(a_range, a, color='b', marker="P", label="$\sigma=10,\mu=0$", linewidth=0.5)
	# bb, = plt.plot(b_range, b, color='g', marker='d', label="$\sigma=5,\mu=0$", linewidth=0.5)
	# # cc, = plt.plot(c_range, c, color='r', marker='^', label="$\sigma=3,\mu=0$", linewidth=0.5)
	# dd, = plt.plot(d_range, d, color='y', marker=".", label="$\sigma=2,\mu=0$", linewidth=0.5)
	ee, = plt.plot(e_range, e, color='k', marker="P", label="$\sigma=0.5,\mu=0$", linewidth=0.5)
	ff, = plt.plot(f_range, f, color='b', marker='.', label="$\sigma=0.4,\mu=0$", linewidth=0.5)
	gg, = plt.plot(g_range, g, color='r', marker='d', label="$\sigma=0.3,\mu=0$", linewidth=0.5)
	hh, = plt.plot(h_range, h, color='c', marker=".", label="$\sigma=0.2,\mu=0$", linewidth=0.5)
	# ii, = plt.plot(j_range, i, color='y', marker="^", label="$\sigma=0.1,\mu=0$", linewidth=0.5)
	# jj, = plt.plot(j_range, j, color='m', marker=".", label="$\sigma=0.15,\mu=0$", linewidth=0.5)
	kk, = plt.plot(k_range, k, color='g', marker="*", label="$\sigma=1.0,\mu=0$", linewidth=0.5)
	
	# plt.legend(handler_map={aa: HandlerLine2D(numpoints=1)})
	# plt.legend([aa, bb, cc, dd], ["Multimodal Uniform ", "Multimodal Gaussian", "Uniform", "Gaussian"],
	#            handler_map={aa: HandlerLine2D(numpoints=1), bb: HandlerLine2D(numpoints=1), cc: HandlerLine2D(numpoints=1),
	#                         dd: HandlerLine2D(numpoints=1)
	
	# }, loc='middle right')
	plt.legend(loc='best')
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy Score")
	# plt.axis("auto")
	plt.grid(True)
	plt.show()
	plt.savefig("Zoom_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler.png")
	plt.close()


# CONFIDENCE
def plot_from_pkl_confidence():
	import numpy as np
	import matplotlib.pyplot as plt
	import pickle
	plt.Figure(figsize=(15, 15))
	
	plt.title('MMInfoGAN confidence by Sampling Method', fontsize=12)
	a = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_MultivariateGaussianSampler_mu_0.0_sigma_0.2_confidence.pkl", "rb"))
	b = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_GaussianSample_mu_0.0_sigma_0.3_confidence.pkl", "rb"))
	c = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_UniformSample_confidence.pkl", "rb"))
	d = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_MultiModalUniformSample_confidence.pkl", "rb"))
	e = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_TruncatedGaussianSample_mu_0.0_sigma_2.0_confidence.pkl", "rb"))
	# plt.plot(a, np.arange(len(a)), 'r--',  b,np.arange(len(b)), 'b--',  c,np.arange(len(c)),'g^',d,np.arange(len(d)),"y--")
	a_range = np.arange(len(a))
	e_range = np.arange(len(e))
	b_range = np.arange(len(b))
	c_range = np.arange(len(c))
	d_range = np.arange(len(d))
	aa, = plt.plot(a_range, a, color='b', marker="P", label="MM Gaussian Sample", linewidth=0.5)
	bb, = plt.plot(b_range, b, color='g', marker='.', label="Gaussian Sample", linewidth=0.5)
	cc, = plt.plot(c_range, c, color='r', marker='^', label="Uniform Sample", linewidth=0.5)
	dd, = plt.plot(d_range, d, color='y', marker="o", label="MM Uniform Sample", linewidth=0.5)
	ee, = plt.plot(e_range, e, color='c', marker="*", label="Truncated Normal Sample", linewidth=0.5)
	mean_line = plt.plot(c_range, np.ones_like(d_range) * 0.98, label='Benchmark', linestyle='--')
	
	# plt.legend(handler_map={aa: HandlerLine2D(numpoints=1)})
	# plt.legend([aa, bb, cc, dd], ["Multimodal Uniform ", "Multimodal Gaussian", "Uniform", "Gaussian"],
	#            handler_map={aa: HandlerLine2D(numpoints=1), bb: HandlerLine2D(numpoints=1), cc: HandlerLine2D(numpoints=1),
	#                         dd: HandlerLine2D(numpoints=1)
	
	# }, loc='middle right')
	plt.legend(loc='best', fancybox=True)
	plt.xlabel("Epoch")
	plt.ylabel("Confidence Score")
	# plt.axis("auto")
	plt.grid(True)
	plt.show()
	plt.savefig("MMInfoGAN confidence by Sampling Method.png")
	plt.close()


def truncated__zoom_plot_from_pkl_confidence():
	import numpy as np
	import matplotlib.pyplot as plt
	import pickle
	plt.Figure(figsize=(15, 15))
	
	plt.title('Zoom_MMinfoGAN_mnist_TruncatedGaussianSample', fontsize=12)
	a = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_TruncatedGaussianSample_mu_0.0_sigma_10.0_confidence.pkl", "rb"))[2:]
	b = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_TruncatedGaussianSample_mu_0.0_sigma_5.0_confidence.pkl", "rb"))[2:]
	c = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_TruncatedGaussianSample_mu_0.0_sigma_3.0_confidence.pkl", "rb"))[2:]
	d = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_TruncatedGaussianSample_mu_0.0_sigma_2.0_confidence.pkl", "rb"))[2:]
	e = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_TruncatedGaussianSample_mu_0.0_sigma_0.5_confidence.pkl", "rb"))[2:]
	f = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_TruncatedGaussianSample_mu_0.0_sigma_0.4_confidence.pkl", "rb"))[2:]
	g = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_TruncatedGaussianSample_mu_0.0_sigma_0.3_confidence.pkl", "rb"))[2:]
	h = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_TruncatedGaussianSample_mu_0.0_sigma_0.2_confidence.pkl", "rb"))[2:]
	# i= pickle.load(open(dir+"classifier_MMinfoGAN_mnist_TruncatedGaussianSample_mu_0.0_sigma_0.1_confidence.pkl", "rb"))
	j = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_TruncatedGaussianSample_mu_0.0_sigma_0.15_confidence.pkl", "rb"))[2:]
	k = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_TruncatedGaussianSample_mu_0.0_sigma_1.0_confidence.pkl", "rb"))[2:]
	# plt.plot(a, np.arange(len(a)), 'r--',  b,np.arange(len(b)), 'b--',  c,np.arange(len(c)),'g^',d,np.arange(len(d)),"y--")
	a_range = np.arange(2, len(a) + 2)
	b_range = np.arange(2, len(b) + 2)
	c_range = np.arange(2, len(c) + 2)
	d_range = np.arange(2, len(d) + 2)
	e_range = np.arange(2, len(e) + 2)
	f_range = np.arange(2, len(f) + 2)
	g_range = np.arange(2, len(g) + 2)
	h_range = np.arange(2, len(h) + 2)
	# i_range = np.arange(len(i))
	j_range = np.arange(len(j))
	k_range = np.arange(len(k))
	aa, = plt.plot(a_range, a, color='b', marker="P", label="$\sigma=10,\mu=0$", linewidth=0.5)
	bb, = plt.plot(b_range, b, color='g', marker='d', label="$\sigma=5,\mu=0$", linewidth=0.5)
	# cc, = plt.plot(c_range, c, color='r', marker='^', label="$\sigma=3,\mu=0$", linewidth=0.5)
	dd, = plt.plot(d_range, d, color='y', marker=".", label="$\sigma=2,\mu=0$", linewidth=0.5)
	ee, = plt.plot(e_range, e, color='k', marker="P", label="$\sigma=0.5,\mu=0$", linewidth=0.5)
	# ff, = plt.plot(f_range, f, color='g', marker='.', label="$\sigma=0.4,\mu=0$", linewidth=0.5)
	# gg, = plt.plot(g_range, g, color='r', marker='>', label="$\sigma=0.3,\mu=0$", linewidth=0.5)
	# hh, = plt.plot(h_range, h, color='c', marker=".", label="$\sigma=0.2,\mu=0$", linewidth=0.5)
	# jj, = plt.plot(j_range, j, color='m', marker=".", label="$\sigma=0.15,\mu=0$", linewidth=0.5)
	# kk, = plt.plot(k_range, k, color='k', marker="*", label="$\sigma=1.0,\mu=0$", linewidth=0.5)
	mean_line = plt.plot(c_range, np.ones_like(d_range) * 0.98, label='Benchmark', linestyle='--')
	
	# plt.legend(handler_map={aa: HandlerLine2D(numpoints=1)})
	# plt.legend([aa, bb, cc, dd], ["Multimodal Uniform ", "Multimodal Gaussian", "Uniform", "Gaussian"],
	#            handler_map={aa: HandlerLine2D(numpoints=1), bb: HandlerLine2D(numpoints=1), cc: HandlerLine2D(numpoints=1),
	#                         dd: HandlerLine2D(numpoints=1)
	
	# }, loc='middle right')
	plt.legend(loc='best')
	plt.xlabel("Epoch")
	plt.ylabel("confidence Score")
	# plt.axis("auto")
	plt.grid(True)
	plt.show()
	plt.savefig("Zoom_MMinfoGAN_mnist_TruncatedGaussianSample.png")
	plt.close()


def truncated_plot_from_pkl_confidence():
	import numpy as np
	import matplotlib.pyplot as plt
	import pickle
	plt.Figure(figsize=(15, 15))
	
	plt.title('MMinfoGAN_mnist_TruncatedGaussianSample', fontsize=12)
	a = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_TruncatedGaussianSample_mu_0.0_sigma_10.0_confidence.pkl", "rb"))
	b = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_TruncatedGaussianSample_mu_0.0_sigma_5.0_confidence.pkl", "rb"))
	c = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_TruncatedGaussianSample_mu_0.0_sigma_3.0_confidence.pkl", "rb"))
	d = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_TruncatedGaussianSample_mu_0.0_sigma_2.0_confidence.pkl", "rb"))
	e = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_TruncatedGaussianSample_mu_0.0_sigma_0.5_confidence.pkl", "rb"))
	f = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_TruncatedGaussianSample_mu_0.0_sigma_0.4_confidence.pkl", "rb"))
	g = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_TruncatedGaussianSample_mu_0.0_sigma_0.3_confidence.pkl", "rb"))
	h = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_TruncatedGaussianSample_mu_0.0_sigma_0.2_confidence.pkl", "rb"))
	# i= pickle.load(open(dir+"classifier_MMinfoGAN_mnist_TruncatedGaussianSample_mu_0.0_sigma_0.1_confidence.pkl", "rb"))
	j = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_TruncatedGaussianSample_mu_0.0_sigma_0.15_confidence.pkl", "rb"))
	k = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_TruncatedGaussianSample_mu_0.0_sigma_1.0_confidence.pkl", "rb"))
	# plt.plot(a, np.arange(len(a)), 'r--',  b,np.arange(len(b)), 'b--',  c,np.arange(len(c)),'g^',d,np.arange(len(d)),"y--")
	a_range = np.arange(len(a))
	b_range = np.arange(len(b))
	c_range = np.arange(len(c))
	d_range = np.arange(len(d))
	e_range = np.arange(len(e))
	f_range = np.arange(len(f))
	g_range = np.arange(len(g))
	h_range = np.arange(len(h))
	# i_range = np.arange(len(i))
	j_range = np.arange(len(j))
	k_range = np.arange(len(k))
	aa, = plt.plot(a_range, a, color='b', marker="P", label="$\sigma=10,\mu=0$", linewidth=0.5)
	# bb, = plt.plot(b_range, b, color='g', marker='d', label="$\sigma=5,\mu=0$", linewidth=0.5)
	# cc, = plt.plot(c_range, c, color='r', marker='^', label="$\sigma=3,\mu=0$", linewidth=0.5)
	dd, = plt.plot(d_range, d, color='y', marker=".", label="$\sigma=2,\mu=0$", linewidth=0.5)
	ee, = plt.plot(e_range, e, color='k', marker="P", label="$\sigma=0.5,\mu=0$", linewidth=0.5)
	ff, = plt.plot(f_range, f, color='g', marker='.', label="$\sigma=0.4,\mu=0$", linewidth=0.5)
	gg, = plt.plot(g_range, g, color='c', marker='>', label="$\sigma=0.3,\mu=0$", linewidth=0.5)
	# hh, = plt.plot(h_range, h, color='c', marker=".", label="$\sigma=0.2,\mu=0$", linewidth=0.5)
	# jj, = plt.plot(j_range, j, color='m', marker=".", label="$\sigma=0.15,\mu=0$", linewidth=0.5)
	kk, = plt.plot(k_range, k, color='g', marker="*", label="$\sigma=1.0,\mu=0$", linewidth=0.5)
	mean_line = plt.plot(c_range, np.ones_like(d_range) * 0.98, label='Benchmark', linestyle='--')
	
	# plt.legend(handler_map={aa: HandlerLine2D(numpoints=1)})
	# plt.legend([aa, bb, cc, dd], ["Multimodal Uniform ", "Multimodal Gaussian", "Uniform", "Gaussian"],
	#            handler_map={aa: HandlerLine2D(numpoints=1), bb: HandlerLine2D(numpoints=1), cc: HandlerLine2D(numpoints=1),
	#                         dd: HandlerLine2D(numpoints=1)
	
	# }, loc='middle right')
	plt.legend(loc='best')
	plt.xlabel("Epoch")
	plt.ylabel("confidence Score")
	# plt.axis("auto")
	plt.grid(True)
	plt.show()
	plt.savefig("MMinfoGAN_mnist_TruncatedGaussianSample.png")
	plt.close()


def gaussian_zoom_plot_from_pkl_confidence():
	import numpy as np
	import matplotlib.pyplot as plt
	import pickle
	plt.Figure(figsize=(15, 15))
	
	plt.title('Zoom_MMinfoGAN_mnist_GaussianSample', fontsize=12)
	# a= pickle.load(open(dir+"classifier_MMinfoGAN_mnist_MultivariateGaussianSample_mu_0.0_sigma_10.0_confidence.pkl", "rb"))[2:]
	# b= pickle.load(open(dir+"classifier_MMinfoGAN_mnist_MultivariateGaussianSample_mu_0.0_sigma_5.0_confidence.pkl", "rb"))[2:]
	# c= pickle.load(open(dir+"classifier_MMinfoGAN_mnist_MultivariateGaussianSample_mu_0.0_sigma_3.0_confidence.pkl", "rb"))[2:]
	# d= pickle.load(open(dir+"classifier_MMinfoGAN_mnist_MultivariateGaussianSample_mu_0.0_sigma_2.0_confidence.pkl", "rb"))[2:]
	e = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_GaussianSample_mu_0.0_sigma_0.5_confidence.pkl", "rb"))[2:]
	f = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_GaussianSample_mu_0.0_sigma_0.4_confidence.pkl", "rb"))[2:]
	g = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_GaussianSample_mu_0.0_sigma_0.3_confidence.pkl", "rb"))[2:]
	h = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_GaussianSample_mu_0.0_sigma_0.2_confidence.pkl", "rb"))[2:]
	i = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_GaussianSample_mu_0.0_sigma_0.1_confidence.pkl", "rb"))[2:]
	j = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_GaussianSample_mu_0.0_sigma_0.15_confidence.pkl", "rb"))[2:]
	k = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_GaussianSample_mu_0.0_sigma_1.0_confidence.pkl", "rb"))[2:]
	# plt.plot(a, np.arange(len(a)), 'r--',  b,np.arange(len(b)), 'b--',  c,np.arange(len(c)),'g^',d,np.arange(len(d)),"y--")
	# a_range = np.arange(2,len(a)+2)
	# b_range = np.arange(2,len(b)+2)
	# c_range = np.arange(2,len(c)+2)
	# d_range = np.arange(2,len(d)+2)
	e_range = np.arange(2, len(e) + 2)
	f_range = np.arange(2, len(f) + 2)
	g_range = np.arange(2, len(g) + 2)
	h_range = np.arange(2, len(h) + 2)
	i_range = np.arange(len(i))
	j_range = np.arange(len(j))
	k_range = np.arange(len(k))
	# aa, = plt.plot(a_range, a, color='b', marker="P", label="$\sigma=10,\mu=0$", linewidth=0.5)
	# bb, = plt.plot(b_range, b, color='g', marker='d', label="$\sigma=5,\mu=0$", linewidth=0.5)
	# # cc, = plt.plot(c_range, c, color='r', marker='^', label="$\sigma=3,\mu=0$", linewidth=0.5)
	# dd, = plt.plot(d_range, d, color='y', marker=".", label="$\sigma=2,\mu=0$", linewidth=0.5)
	ee, = plt.plot(e_range, e, color='k', marker="P", label="$\sigma=0.5,\mu=0$", linewidth=0.5)
	ff, = plt.plot(f_range, f, color='b', marker='.', label="$\sigma=0.4,\mu=0$", linewidth=0.5)
	# gg, = plt.plot(g_range, g, color='r', marker='d', label="$\sigma=0.3,\mu=0$", linewidth=0.5)
	hh, = plt.plot(h_range, h, color='c', marker=".", label="$\sigma=0.2,\mu=0$", linewidth=0.5)
	ii, = plt.plot(j_range, i, color='y', marker="^", label="$\sigma=0.1,\mu=0$", linewidth=0.5)
	jj, = plt.plot(j_range, j, color='m', marker=".", label="$\sigma=0.15,\mu=0$", linewidth=0.5)
	kk, = plt.plot(k_range, k, color='g', marker="*", label="$\sigma=1.0,\mu=0$", linewidth=0.5)
	
	# plt.legend(handler_map={aa: HandlerLine2D(numpoints=1)})
	# plt.legend([aa, bb, cc, dd], ["Multimodal Uniform ", "Multimodal Gaussian", "Uniform", "Gaussian"],
	#            handler_map={aa: HandlerLine2D(numpoints=1), bb: HandlerLine2D(numpoints=1), cc: HandlerLine2D(numpoints=1),
	#                         dd: HandlerLine2D(numpoints=1)
	
	# }, loc='middle right')
	plt.legend(loc='best')
	plt.xlabel("Epoch")
	plt.ylabel("confidence Score")
	# plt.axis("auto")
	plt.grid(True)
	plt.show()
	plt.savefig("Zoom_MMinfoGAN_mnist_GaussianSample.png")
	plt.close()


def gaussian_plot_from_pkl_confidence():
	import numpy as np
	import matplotlib.pyplot as plt
	import pickle
	plt.Figure(figsize=(15, 15))
	
	plt.title('MMinfoGAN_mnist_GaussianSample', fontsize=12)
	# a= pickle.load(open(dir+"classifier_MMinfoGAN_mnist_MultivariateGaussianSample_mu_0.0_sigma_10.0_confidence.pkl", "rb"))
	# b= pickle.load(open(dir+"classifier_MMinfoGAN_mnist_MultivariateGaussianSample_mu_0.0_sigma_5.0_confidence.pkl", "rb"))
	# c= pickle.load(open(dir+"classifier_MMinfoGAN_mnist_MultivariateGaussianSample_mu_0.0_sigma_3.0_confidence.pkl", "rb"))
	# d= pickle.load(open(dir+"classifier_MMinfoGAN_mnist_MultivariateGaussianSample_mu_0.0_sigma_2.0_confidence.pkl", "rb"))
	e = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_GaussianSample_mu_0.0_sigma_0.5_confidence.pkl", "rb"))
	f = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_GaussianSample_mu_0.0_sigma_0.4_confidence.pkl", "rb"))
	g = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_GaussianSample_mu_0.0_sigma_0.3_confidence.pkl", "rb"))
	h = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_GaussianSample_mu_0.0_sigma_0.2_confidence.pkl", "rb"))
	i = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_GaussianSample_mu_0.0_sigma_0.1_confidence.pkl", "rb"))
	j = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_GaussianSample_mu_0.0_sigma_0.15_confidence.pkl", "rb"))
	k = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_GaussianSample_mu_0.0_sigma_1.0_confidence.pkl", "rb"))
	# a_range = np.arange(len(a))
	# b_range = np.arange(len(b))
	# c_range = np.arange(len(c))
	# d_range = np.arange(len(d))
	e_range = np.arange(len(e))
	f_range = np.arange(len(f))
	g_range = np.arange(len(g))
	h_range = np.arange(len(h))
	i_range = np.arange(len(i))
	j_range = np.arange(len(j))
	k_range = np.arange(len(k))
	# aa, = plt.plot(a_range, a, color='b', marker="P", label="$\sigma=10,\mu=0$", linewidth=0.5)
	# bb, = plt.plot(b_range, b, color='g', marker='d', label="$\sigma=5,\mu=0$", linewidth=0.5)
	# cc, = plt.plot(c_range, c, color='r', marker='^', label="$\sigma=3,\mu=0$", linewidth=0.5)
	# dd, = plt.plot(d_range, d, color='y', marker=".", label="$\sigma=2,\mu=0$", linewidth=0.5)
	ee, = plt.plot(e_range, e, color='k', marker="P", label="$\sigma=0.5,\mu=0$", linewidth=0.5)
	# ff, = plt.plot(f_range, f, color='b', marker='.', label="$\sigma=0.4,\mu=0$", linewidth=0.5)
	gg, = plt.plot(g_range, g, color='r', marker='d', label="$\sigma=0.3,\mu=0$", linewidth=0.5)
	hh, = plt.plot(h_range, h, color='c', marker=".", label="$\sigma=0.2,\mu=0$", linewidth=0.5)
	jj, = plt.plot(j_range, j, color='m', marker=".", label="$\sigma=0.1,\mu=0$", linewidth=0.5)
	ii, = plt.plot(j_range, i, color='y', marker="^", label="$\sigma=0.15,\mu=0$", linewidth=0.5)
	kk, = plt.plot(k_range, k, color='g', marker="*", label="$\sigma=1.0,\mu=0$", linewidth=0.5)
	mean_line = plt.plot(e_range, np.ones_like(e_range) * 0.98, label='Benchmark', linestyle='--')
	
	# plt.legend(handler_map={aa: HandlerLine2D(numpoints=1)})
	# plt.legend([aa, bb, cc, dd], ["Multimodal Uniform ", "Multimodal Gaussian", "Uniform", "Gaussian"],
	#            handler_map={aa: HandlerLine2D(numpoints=1), bb: HandlerLine2D(numpoints=1), cc: HandlerLine2D(numpoints=1),
	#                         dd: HandlerLine2D(numpoints=1)
	
	# }, loc='middle right')
	plt.legend(loc='best')
	plt.xlabel("Epoch")
	plt.ylabel("confidence Score")
	# plt.axis("auto")
	plt.grid(True)
	plt.show()
	plt.savefig("MMinfoGAN_mnist_GaussianSampler.png")
	plt.close()


def MM_zoom_plot_from_pkl_confidence():
	import numpy as np
	import matplotlib.pyplot as plt
	import pickle
	plt.Figure(figsize=(15, 15))
	
	plt.title('Zoom_MMinfoGAN_mnist_MultivariateGaussianSampler', fontsize=12)
	# a= pickle.load(open(dir+"classifier_MMinfoGAN_mnist_MultivariateMultivariateGaussianSampler_mu_0.0_sigma_10.0_confidence.pkl", "rb"))[2:]
	# b= pickle.load(open(dir+"classifier_MMinfoGAN_mnist_MultivariateMultivariateGaussianSampler_mu_0.0_sigma_5.0_confidence.pkl", "rb"))[2:]
	# c= pickle.load(open(dir+"classifier_MMinfoGAN_mnist_MultivariateMultivariateGaussianSampler_mu_0.0_sigma_3.0_confidence.pkl", "rb"))[2:]
	# d= pickle.load(open(dir+"classifier_MMinfoGAN_mnist_MultivariateMultivariateGaussianSampler_mu_0.0_sigma_2.0_confidence.pkl", "rb"))[2:]
	e = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_MultivariateGaussianSampler_mu_0.0_sigma_0.5_confidence.pkl", "rb"))[2:]
	f = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_MultivariateGaussianSampler_mu_0.0_sigma_0.4_confidence.pkl", "rb"))[2:]
	g = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_MultivariateGaussianSampler_mu_0.0_sigma_0.3_confidence.pkl", "rb"))[2:]
	h = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_MultivariateGaussianSampler_mu_0.0_sigma_0.2_confidence.pkl", "rb"))[2:]
	i = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_MultivariateGaussianSampler_mu_0.0_sigma_0.1_confidence.pkl", "rb"))[2:]
	j = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_MultivariateGaussianSampler_mu_0.0_sigma_0.15_confidence.pkl", "rb"))[2:]
	k = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_MultivariateGaussianSampler_mu_0.0_sigma_1.0_confidence.pkl", "rb"))[2:]
	# plt.plot(a, np.arange(len(a)), 'r--',  b,np.arange(len(b)), 'b--',  c,np.arange(len(c)),'g^',d,np.arange(len(d)),"y--")
	# a_range = np.arange(2,len(a)+2)
	# b_range = np.arange(2,len(b)+2)
	# c_range = np.arange(2,len(c)+2)
	# d_range = np.arange(2,len(d)+2)
	e_range = np.arange(2, len(e) + 2)
	f_range = np.arange(2, len(f) + 2)
	g_range = np.arange(2, len(g) + 2)
	h_range = np.arange(2, len(h) + 2)
	i_range = np.arange(len(i))
	j_range = np.arange(len(j))
	k_range = np.arange(len(k))
	# aa, = plt.plot(a_range, a, color='b', marker="P", label="$\sigma=10,\mu=0$", linewidth=0.5)
	# bb, = plt.plot(b_range, b, color='g', marker='d', label="$\sigma=5,\mu=0$", linewidth=0.5)
	# # cc, = plt.plot(c_range, c, color='r', marker='^', label="$\sigma=3,\mu=0$", linewidth=0.5)
	# dd, = plt.plot(d_range, d, color='y', marker=".", label="$\sigma=2,\mu=0$", linewidth=0.5)
	ee, = plt.plot(e_range, e, color='k', marker="P", label="$\sigma=0.5,\mu=0$", linewidth=0.5)
	ff, = plt.plot(f_range, f, color='b', marker='.', label="$\sigma=0.4,\mu=0$", linewidth=0.5)
	gg, = plt.plot(g_range, g, color='r', marker='d', label="$\sigma=0.3,\mu=0$", linewidth=0.5)
	hh, = plt.plot(h_range, h, color='c', marker=".", label="$\sigma=0.2,\mu=0$", linewidth=0.5)
	# ii, = plt.plot(j_range, i, color='y', marker="^", label="$\sigma=0.1,\mu=0$", linewidth=0.5)
	# jj, = plt.plot(j_range, j, color='m', marker=".", label="$\sigma=0.15,\mu=0$", linewidth=0.5)
	kk, = plt.plot(k_range, k, color='g', marker="*", label="$\sigma=1.0,\mu=0$", linewidth=0.5)
	
	# plt.legend(handler_map={aa: HandlerLine2D(numpoints=1)})
	# plt.legend([aa, bb, cc, dd], ["Multimodal Uniform ", "Multimodal Gaussian", "Uniform", "Gaussian"],
	#            handler_map={aa: HandlerLine2D(numpoints=1), bb: HandlerLine2D(numpoints=1), cc: HandlerLine2D(numpoints=1),
	#                         dd: HandlerLine2D(numpoints=1)
	
	# }, loc='middle right')
	plt.legend(loc='best')
	plt.xlabel("Epoch")
	plt.ylabel("confidence Score")
	# plt.axis("auto")
	plt.grid(True)
	plt.show()
	plt.savefig("Zoom_MMinfoGAN_mnist_MultivariateGaussianSampler.png")
	plt.close()


def MM_plot_from_pkl_confidence():
	import numpy as np
	import matplotlib.pyplot as plt
	import pickle
	plt.Figure(figsize=(15, 15))
	
	plt.title('MMinfoGAN_mnist_MultivariateGaussianSampler', fontsize=12)
	# a= pickle.load(open(dir+"classifier_MMinfoGAN_mnist_MultivariateMultivariateGaussianSampler_mu_0.0_sigma_10.0_confidence.pkl", "rb"))
	# b= pickle.load(open(dir+"classifier_MMinfoGAN_mnist_MultivariateMultivariateGaussianSampler_mu_0.0_sigma_5.0_confidence.pkl", "rb"))
	# c= pickle.load(open(dir+"classifier_MMinfoGAN_mnist_MultivariateMultivariateGaussianSampler_mu_0.0_sigma_3.0_confidence.pkl", "rb"))
	# d= pickle.load(open(dir+"classifier_MMinfoGAN_mnist_MultivariateMultivariateGaussianSampler_mu_0.0_sigma_2.0_confidence.pkl", "rb"))
	e = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_MultivariateGaussianSampler_mu_0.0_sigma_0.5_confidence.pkl", "rb"))
	f = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_MultivariateGaussianSampler_mu_0.0_sigma_0.4_confidence.pkl", "rb"))
	g = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_MultivariateGaussianSampler_mu_0.0_sigma_0.3_confidence.pkl", "rb"))
	h = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_MultivariateGaussianSampler_mu_0.0_sigma_0.2_confidence.pkl", "rb"))
	i = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_MultivariateGaussianSampler_mu_0.0_sigma_0.1_confidence.pkl", "rb"))
	j = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_MultivariateGaussianSampler_mu_0.0_sigma_0.15_confidence.pkl", "rb"))
	k = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_MultivariateGaussianSampler_mu_0.0_sigma_1.0_confidence.pkl", "rb"))
	# a_range = np.arange(len(a))
	# b_range = np.arange(len(b))
	# c_range = np.arange(len(c))
	# d_range = np.arange(len(d))
	e_range = np.arange(len(e))
	f_range = np.arange(len(f))
	g_range = np.arange(len(g))
	h_range = np.arange(len(h))
	i_range = np.arange(len(i))
	j_range = np.arange(len(j))
	k_range = np.arange(len(k))
	# aa, = plt.plot(a_range, a, color='b', marker="P", label="$\sigma=10,\mu=0$", linewidth=0.5)
	# bb, = plt.plot(b_range, b, color='g', marker='d', label="$\sigma=5,\mu=0$", linewidth=0.5)
	# cc, = plt.plot(c_range, c, color='r', marker='^', label="$\sigma=3,\mu=0$", linewidth=0.5)
	# dd, = plt.plot(d_range, d, color='y', marker=".", label="$\sigma=2,\mu=0$", linewidth=0.5)
	ee, = plt.plot(e_range, e, color='k', marker="P", label="$\sigma=0.5,\mu=0$", linewidth=0.5)
	ff, = plt.plot(f_range, f, color='b', marker='.', label="$\sigma=0.4,\mu=0$", linewidth=0.5)
	gg, = plt.plot(g_range, g, color='r', marker='d', label="$\sigma=0.3,\mu=0$", linewidth=0.5)
	hh, = plt.plot(h_range, h, color='c', marker=".", label="$\sigma=0.2,\mu=0$", linewidth=0.5)
	# jj, = plt.plot(j_range, j, color='m', marker=".", label="$\sigma=0.1,\mu=0$", linewidth=0.5)
	ii, = plt.plot(j_range, i, color='y', marker="^", label="$\sigma=0.15,\mu=0$", linewidth=0.5)
	kk, = plt.plot(k_range, k, color='g', marker="*", label="$\sigma=1.0,\mu=0$", linewidth=0.5)
	mean_line = plt.plot(e_range, np.ones_like(e_range) * 0.98, label='Benchmark', linestyle='--')
	
	# plt.legend(handler_map={aa: HandlerLine2D(numpoints=1)})
	# plt.legend([aa, bb, cc, dd], ["Multimodal Uniform ", "Multimodal Gaussian", "Uniform", "Gaussian"],
	#            handler_map={aa: HandlerLine2D(numpoints=1), bb: HandlerLine2D(numpoints=1), cc: HandlerLine2D(numpoints=1),
	#                         dd: HandlerLine2D(numpoints=1)
	
	# }, loc='middle right')
	plt.legend(loc='best')
	plt.xlabel("Epoch")
	plt.ylabel("confidence Score")
	# plt.axis("auto")
	plt.grid(True)
	plt.show()
	plt.savefig("MMinfoGAN_mnist_MultivariateGaussianSampler.png")
	plt.close()


if __name__ == '__main__':
	fashion_MM_plot_from_pkl()
	# plot_from_pkl()
	# MM_mu_1_zoom_plot_from_pkl()
	# MM_mu_05_07_08_zoom_plot_from_pkl()  # MM_mu_01_zoom_plot_from_pkl()
