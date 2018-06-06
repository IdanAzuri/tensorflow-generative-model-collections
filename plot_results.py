import sys

import matplotlib

matplotlib.use('Agg')
def plot_from_pkl():
	import numpy as np
	import matplotlib.pyplot as plt
	import pickle
	plt.Figure(figsize=(15, 15))
	dir = 'classifier_results/'
	plt.title('MMInfoGAN Accuracy by Sampling Method', fontsize=12)
	a = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_MultivariateGaussianSampler_mu_0.0_sigma_0.2_accuracy.pkl", "rb"))
	b = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_GaussianSample_mu_0.0_sigma_0.3_accuracy.pkl", "rb"))
	c = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_UniformSample_accuracy.pkl", "rb"))
	d = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_MultiModalUniformSample_accuracy.pkl", "rb"))
	e = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_TruncatedGaussianSample_mu_0.0_sigma_2.0_accuracy.pkl", "rb"))
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
	mean_line = plt.plot(c_range, np.ones_like(d_range) * 0.92, label='Benchmark', linestyle='--')

	# plt.legend(handler_map={aa: HandlerLine2D(numpoints=1)})
	# plt.legend([aa, bb, cc, dd], ["Multimodal Uniform ", "Multimodal Gaussian", "Uniform", "Gaussian"],
	#            handler_map={aa: HandlerLine2D(numpoints=1), bb: HandlerLine2D(numpoints=1), cc: HandlerLine2D(numpoints=1),
	#                         dd: HandlerLine2D(numpoints=1)

	# }, loc='middle right')
	plt.legend(loc='best')
	plt.xlabel("Epoch")
	plt.ylabel("Confidence Score")
	# plt.axis("auto")
	plt.grid(True)
	plt.show()
	plt.savefig("MMInfoGAN Accuracy by Sampling Method.png")
	plt.close()

#ACCURACY
def truncated__zoom_plot_from_pkl():
	import numpy as np
	import matplotlib.pyplot as plt
	import pickle
	plt.Figure(figsize=(15, 15))
	dir = 'classifier_results/'
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
	mean_line = plt.plot(c_range, np.ones_like(d_range) * 0.92, label='Benchmark', linestyle='--')

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
	dir = 'classifier_results/'
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
	mean_line = plt.plot(c_range, np.ones_like(d_range) * 0.92, label='Benchmark', linestyle='--')

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
	dir = 'classifier_results/'
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
	dir = 'classifier_results/'
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
	mean_line = plt.plot(e_range, np.ones_like(e_range) * 0.92, label='Benchmark', linestyle='--')

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
	dir = 'classifier_results/'
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
	dir = 'classifier_results/'
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
	mean_line = plt.plot(e_range, np.ones_like(e_range) * 0.92, label='Benchmark', linestyle='--')

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
def MM_mu_1_zoom_plot_from_pkl():
	import numpy as np
	import matplotlib.pyplot as plt
	import pickle
	plt.Figure(figsize=(15, 15))
	dir = 'classifier_results/'
	plt.title('Zoom_MMinfoGAN_mnist_MultivariateGaussianSampler $\mu=0.1$', fontsize=12)
	# a= pickle.load(open(dir+"classifier_MMinfoGAN_mnist_MultivariateMultivariateGaussianSampler_mu_0.1_sigma_10.0_accuracy.pkl", "rb"))[2:]
	# b= pickle.load(open(dir+"classifier_MMinfoGAN_mnist_MultivariateMultivariateGaussianSampler_mu_0.1_sigma_5.0_accuracy.pkl", "rb"))[2:]
	# c= pickle.load(open(dir+"classifier_MMinfoGAN_mnist_MultivariateMultivariateGaussianSampler_mu_0.1_sigma_3.0_accuracy.pkl", "rb"))[2:]
	# d= pickle.load(open(dir+"classifier_MMinfoGAN_mnist_MultivariateMultivariateGaussianSampler_mu_0.1_sigma_2.0_accuracy.pkl", "rb"))[2:]
	e = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.5_accuracy.pkl", "rb"))[2:]
	f = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.4_accuracy.pkl", "rb"))[2:]
	g = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.3_accuracy.pkl", "rb"))[2:]
	h = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.2_accuracy.pkl", "rb"))[2:]
	i = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.1_accuracy.pkl", "rb"))[2:]
	j = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_MultivariateGaussianSampler_mu_0.1_sigma_0.15_accuracy.pkl", "rb"))[2:]
	k = pickle.load(open(dir + "classifier_MMinfoGAN_mnist_MultivariateGaussianSampler_mu_0.1_sigma_1.0_accuracy.pkl", "rb"))[2:]
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
	plt.savefig("Zoom_MMinfoGAN_mnist_MultivariateGaussianSampler $\mu=0.1$.png")
	plt.close()


def MM_mu_1plot_from_pkl():
	import numpy as np
	import matplotlib.pyplot as plt
	import pickle
	plt.Figure(figsize=(15, 15))
	dir = 'classifier_results/'
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
	mean_line = plt.plot(e_range, np.ones_like(e_range) * 0.92, label='Benchmark', linestyle='--')

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




#ACCURACY fashion
def truncated_fashion__zoom_plot_from_pkl():
	import numpy as np
	import matplotlib.pyplot as plt
	import pickle
	plt.Figure(figsize=(15, 15))
	dir = 'classifier_results/'
	plt.title('Zoom_MMinfoGAN_fashion-mnist_TruncatedGaussianSample', fontsize=12)
	a = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_TruncatedGaussianSample_mu_0.0_sigma_10.0_accuracy.pkl", "rb"))[2:]
	b = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_TruncatedGaussianSample_mu_0.0_sigma_5.0_accuracy.pkl", "rb"))[2:]
	c = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_TruncatedGaussianSample_mu_0.0_sigma_3.0_accuracy.pkl", "rb"))[2:]
	d = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_TruncatedGaussianSample_mu_0.0_sigma_2.0_accuracy.pkl", "rb"))[2:]
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
	mean_line = plt.plot(c_range, np.ones_like(d_range) * 0.92, label='Benchmark', linestyle='--')

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
	dir = 'classifier_results/'
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
	mean_line = plt.plot(c_range, np.ones_like(d_range) * 0.92, label='Benchmark', linestyle='--')

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
	dir = 'classifier_results/'
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
	dir = 'classifier_results/'
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
	mean_line = plt.plot(e_range, np.ones_like(e_range) * 0.92, label='Benchmark', linestyle='--')

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
	dir = 'classifier_results/'
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


def fashion_MM_plot_from_pkl():
	import numpy as np
	import matplotlib.pyplot as plt
	import pickle
	plt.Figure(figsize=(15, 15))
	dir = 'classifier_results/'
	plt.title('MMinfoGAN_fashion-mnist_MultivariateGaussianSampler', fontsize=12)
	# a= pickle.load(open(dir+"classifier_MMinfoGAN_fashion-mnist_MultivariateMultivariateGaussianSampler_mu_0.0_sigma_10.0_accuracy.pkl", "rb"))
	# b= pickle.load(open(dir+"classifier_MMinfoGAN_fashion-mnist_MultivariateMultivariateGaussianSampler_mu_0.0_sigma_5.0_accuracy.pkl", "rb"))
	# c= pickle.load(open(dir+"classifier_MMinfoGAN_fashion-mnist_MultivariateMultivariateGaussianSampler_mu_0.0_sigma_3.0_accuracy.pkl", "rb"))
	# d= pickle.load(open(dir+"classifier_MMinfoGAN_fashion-mnist_MultivariateMultivariateGaussianSampler_mu_0.0_sigma_2.0_accuracy.pkl", "rb"))
	e = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.0_sigma_0.5_accuracy.pkl", "rb"))
	f = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.0_sigma_0.4_accuracy.pkl", "rb"))
	g = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.0_sigma_0.3_accuracy.pkl", "rb"))
	h = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.0_sigma_0.2_accuracy.pkl", "rb"))
	i = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.0_sigma_0.1_accuracy.pkl", "rb"))
	j = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.0_sigma_0.15_accuracy.pkl", "rb"))
	k = pickle.load(open(dir + "classifier_MMinfoGAN_fashion-mnist_MultivariateGaussianSampler_mu_0.0_sigma_1.0_accuracy.pkl", "rb"))
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
	mean_line = plt.plot(e_range, np.ones_like(e_range) * 0.92, label='Benchmark', linestyle='--')

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
	plt.savefig("MMinfoGAN_fashion-mnist_MultivariateGaussianSampler.png")
	plt.close()







#CONFIDENCE
def plot_from_pkl_confidence():
	import numpy as np
	import matplotlib.pyplot as plt
	import pickle
	plt.Figure(figsize=(15, 15))
	dir = 'classifier_results/'
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
	mean_line = plt.plot(c_range, np.ones_like(d_range) * 0.92, label='Benchmark', linestyle='--')

	# plt.legend(handler_map={aa: HandlerLine2D(numpoints=1)})
	# plt.legend([aa, bb, cc, dd], ["Multimodal Uniform ", "Multimodal Gaussian", "Uniform", "Gaussian"],
	#            handler_map={aa: HandlerLine2D(numpoints=1), bb: HandlerLine2D(numpoints=1), cc: HandlerLine2D(numpoints=1),
	#                         dd: HandlerLine2D(numpoints=1)

	# }, loc='middle right')
	plt.legend(loc='best')
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
	dir = 'classifier_results/'
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
	mean_line = plt.plot(c_range, np.ones_like(d_range) * 0.92, label='Benchmark', linestyle='--')

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
	dir = 'classifier_results/'
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
	mean_line = plt.plot(c_range, np.ones_like(d_range) * 0.92, label='Benchmark', linestyle='--')

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
	dir = 'classifier_results/'
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
	dir = 'classifier_results/'
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
	mean_line = plt.plot(e_range, np.ones_like(e_range) * 0.92, label='Benchmark', linestyle='--')

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
	dir = 'classifier_results/'
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
	dir = 'classifier_results/'
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
	mean_line = plt.plot(e_range, np.ones_like(e_range) * 0.92, label='Benchmark', linestyle='--')

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
	# gaussian_plot_from_pkl_confidence()
	# gaussian_zoom_plot_from_pkl_confidence()
	# plot_from_pkl_confidence()
	# plot_from_pkl_confidence()
	# truncated__zoom_plot_from_pkl_confidence()
	# truncated_plot_from_pkl_confidence()
	# MM_plot_from_pkl_confidence()
	# MM_zoom_plot_from_pkl_confidence()

	# fashion_gaussian_plot_from_pkl()
	# fashion_gaussian_zoom_plot_from_pkl()
	# fashion_MM_plot_from_pkl()
	# fashion_MM_zoom_plot_from_pkl()
	# fashion_truncated_plot_from_pkl()
	# truncated_fashion__zoom_plot_from_pkl()

	# MM_mu_1plot_from_pkl()
	# MM_mu_1_zoom_plot_from_pkl()
	plot_from_pkl()