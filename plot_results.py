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
START = 3
start = START
END = 50
# regex
# (classifier_MM.*_sigma_\d.\d)(.*)(_ndist_\d+)(_accuracy)(.pkl)
# $1$3$4_cv_3_gan_only_no_prior$5



def MMgeneral_plot_from_pkl(groupby=""):
	import glob, os
	param_list = []
	files_list = defaultdict(list)
	dirs = [d for d in glob.iglob("/Users/idan.a/repos/tensorflow-generative-model-collections/classifier_results_seed_*")]
	
	for dir in dirs:
		for f in glob.iglob("{}/*{}*.pkl".format(dir,groupby)):
			fname=f.split("/")[-1]
			tmp=fname.split("_")
			mu=tmp[5]
			sigma=tmp[7]
			ndist=tmp[9]
			param_list.append("$\Sigma={},\mu={}$".format(sigma,mu))
			print(fname,f)
			try:
				np_max = np.max(pickle.load(open(f, "rb")))
				# np_max = pickle.load(open(f, "rb"))[-1]
				files_list[fname].append(np_max)
			except Exception as e:
				print("ERROR:{}\n{}".format(f,e))
				
				
	
	
	means=[]
	std_errs=[]
	for key in files_list.keys():
		current_experiment = files_list[key]
		means.append(np.mean(current_experiment, axis=0))
		std_errs.append(np.std(current_experiment, axis=0) / len(current_experiment))
	



	fig, ax = plt.subplots()
	models = set(param_list)
	title = 'MMinfoGAN_Fsion-Mnist_multi-modal Multi modal Gaussian - {} modals'.format(groupby)
	
	ax.set_title(title, fontsize=10)
	x_pos = np.arange(len(models))
	ax.bar(x_pos, means, yerr=std_errs, align='center', alpha=0.5, ecolor='black', capsize=10)
	ax.set_ylabel('Accuracy')
	ax.set_xticks(x_pos)
	ax.set_xticklabels(models)
	plt.xticks(rotation=90)
	ax.set_ylim([0.5,0.63])
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
	# fashion_MM_plot_from_pkl()
	# # plot_from_pkl()
	# MM_mu_1_zoom_plot_from_pkl()
	# MM_mu_01_zoom_plot_from_pkl()
	# MM_mu_05_07_08_zoom_plot_from_pkl()
	# MM_mu_05_07_08_zoom_plot_from_pkl_5_modals()
	# MM_mu_1_zoom_plot_from_pkl_5modals()
	# MM_mu_01_zoom_plot_from_pkl_5modals()
	# MM_mu_1_zoom_plot_from_pkl3modals()
	# MM_mu_01_zoom_plot_from_pkl3modals()
	# MM_mu_05_07_08_zoom_plot_from_pkl3modals()
	MMgeneral_plot_from_pkl("GaussianSample_")
	MMgeneral_plot_from_pkl("Uniform")
	MMgeneral_plot_from_pkl("MultivariateGaussianSampler*ndist_10")
	MMgeneral_plot_from_pkl("MultivariateGaussianSampler*ndist_5")
	MMgeneral_plot_from_pkl("MultivariateGaussianSampler*ndist_3")
