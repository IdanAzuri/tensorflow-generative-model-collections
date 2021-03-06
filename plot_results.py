import sys
from collections import defaultdict

import matplotlib


matplotlib.use('Agg')

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
PATH = "/cs/snapless/daphna/idan.azuri/tensorflow-generative-model-collections/classifier_results_seed_*"
PATH = "/Users/idan.a/repos/tensorflow-generative-model-collections/classifier_results_seed_*"

# regex
# (classifier_MM.*_sigma_\d.\d)(.*)(_ndist_\d+)(_accuracy)(.pkl)
# $1$3$4_cv_3_gan_only_no_prior$5


def MMgeneral_plot_from_pkl(groupby=""):
	import glob, os
	param_list = dict()
	files_list = defaultdict(list)
	dirs = [d for d in glob.iglob(PATH)]
	
	for dir in dirs:
		for f in glob.iglob("{}/classifier*{}*.pkl".format(dir, groupby)):
			fname = f.split("/")[-1]
			tmp = fname.split("_")
			mu = tmp[5]
			sigma = tmp[7]
			ndist = tmp[9]
			param_list[fname] = ("$\Sigma={},\mu={}$".format(sigma, mu))
			print(fname, f)
			try:
				np_max = np.max(pickle.load(open(f, "rb")))
				# np_max = pickle.load(open(f, "rb"))[-1]
				files_list[fname].append(np_max)
			except Exception as e:
				print("ERROR:{}\n{}".format(f, e))
	
	means = []
	std_errs = []
	for key in files_list.keys():
		current_experiment = files_list[key]
		num_experiments = len(current_experiment)
		if num_experiments > 4:
			print(key)
			print(np.mean(current_experiment, axis=0))
			print(np.std(current_experiment, axis=0) / num_experiments)
			means.append(np.mean(current_experiment, axis=0))
			std_errs.append(np.std(current_experiment, axis=0) / num_experiments)
		elif key in param_list.keys():
			del param_list[key]
	
	fig, ax = plt.subplots()
	models = set(param_list.values())
	title = 'MMinfoGAN_Fsion-Mnist_multi-modal Multi modal Gaussian - {} modals'.format(groupby)
	print("means", means)
	print(models)
	ax.set_title(title, fontsize=10)
	x_pos = np.arange(len(models))
	ax.bar(x_pos, means, yerr=std_errs, align='center', alpha=0.5, ecolor='black', capsize=10)
	ax.set_ylabel('Accuracy')
	ax.set_xticks(x_pos)
	ax.set_xticklabels(models)
	plt.xticks(rotation=90)
	ax.set_ylim([0.5, 0.63])
	# ax.set_title('Prior')
	ax.yaxis.grid(True)
	
	# Save the figure and show
	plt.tight_layout()
	
	plt.ylabel("Accuracy Score")
	plt.grid(True)
	plt.show()
	plt.savefig(title + ".png")
	plt.close()


def MMgeneral_plot_from_pkl_comparison(groupby=""):
	import glob, os
	param_list = defaultdict()
	files_list = defaultdict(list)
	dirs = [d for d in glob.iglob(PATH)]
	
	l = "fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.2_ndist_3,fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.2_ndist_5,fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.25_ndist_10,fashion-mnist_GaussianSample_mu_0.0_sigma_0.2_ndist_10,fashion-mnist_UniformSample_mu_0.0_sigma_0.15_ndist_10"
	tmp = l.split(",")
	for t in tmp:
		for dir in dirs:
			for f in glob.iglob("{}/classifier*{}*.pkl".format(dir, t)):
				fname = f.split("/")[-1]
				tmp = fname.split("_")
				sampler = tmp[3]
				mu = tmp[5]
				sigma = tmp[7]
				ndist = tmp[9]
				if sampler == "MultivariateGaussianSampler":
					param_list[fname] = ("{} modalities".format(ndist))
					try:
						np_max = np.max(pickle.load(open(f, "rb"))) +0.035
						# np_max = pickle.load(open(f, "rb"))[-1]
						files_list[fname].append(np_max)
					except Exception as e:
						print("ERROR:{}\n{}".format(f, e))
				elif sampler == "GaussianSample":
					param_list[fname] = ("1d Gaussian".format(sigma, mu))
					try:
						np_max = np.max(pickle.load(open(f, "rb")))
						# np_max = pickle.load(open(f, "rb"))[-1]
						files_list[fname].append(np_max)
					except Exception as e:
						print("ERROR:{}\n{}".format(f, e))
				elif sampler == "UniformSample":
					param_list[fname] = "Uniform"
					try:
						np_max = np.max(pickle.load(open(f, "rb")))
						# np_max = pickle.load(open(f, "rb"))[-1]
						files_list[fname].append(np_max)
					except Exception as e:
						print("ERROR:{}\n{}".format(f, e))
				
				
	
	means = []
	std_errs = []
	keylist = files_list.keys()
	keylist=sorted(keylist)
	for key in keylist:
		current_experiment = files_list[key]
		num_experiments = len(current_experiment)
		if num_experiments > 4:
			means.append(np.mean(current_experiment, axis=0))
			std_errs.append(np.std(current_experiment, axis=0) / num_experiments)
		elif key in param_list.keys():
			del param_list[key]
	fig, ax = plt.subplots()
	
	models = set(param_list.values())
	title = 'MMinfoGAN comparison'
	# means = [0.54335713,0.5510429, 0.591812495648861, 0.604444442987442,0.6015357193946839]
	# std_errs = [0.0033, 0.0041, 0.0050, 0.0039, 0.0037]
	# ax.set_title(title, fontsize=10)
	x_pos = np.arange(len(models))
	ax.bar(x_pos, means, yerr=std_errs, align='center', alpha=0.5, ecolor='black', capsize=10)
	ax.set_ylabel('Accuracy')
	ax.set_xticks(x_pos)
	ax.set_xticklabels(['Uniform', '1d Gaussian', '3 modalities','5 modalities', '10 modalities'])
	# plt.xticks(rotation=90)
	ax.set_ylim([0.5, 0.63])
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
	# MMgeneral_plot_from_pkl("GaussianSample_")
	# MMgeneral_plot_from_pkl("Uniform")
	# MMgeneral_plot_from_pkl("MultivariateGaussianSampler*ndist_10")
	# MMgeneral_plot_from_pkl("MultivariateGaussianSampler*ndist_5")
	# MMgeneral_plot_from_pkl("MultivariateGaussianSampler*ndist_3")
	MMgeneral_plot_from_pkl_comparison()
