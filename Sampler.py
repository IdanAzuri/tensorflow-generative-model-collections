import numpy as np


class Sampler(object):
	def __init__(self):
		pass

	def get_sample(self, dimension, batch_size, n_distributions):
		pass


class MultivariateSamplerSmallVariance(Sampler):
	def get_sample(self, batch_size, embedding_dim, n_distributions):
		current_dist_states_indices = np.random.randint(0, n_distributions - 1, batch_size)
		mean_vec = np.random.randint(low=0, high=20, size=n_distributions)
		cov_mat = np.eye(n_distributions) * 1  # np.random.randint(1, 5, n_distributions)  # this is diagonal beacuse we want iid

		result_vec = np.zeros((batch_size, embedding_dim))
		# create multimodal matrix
		matrix_sample = np.random.multivariate_normal(mean_vec, cov_mat, size=batch_size * embedding_dim)
		# matrix_sample from the multimodal matrix
		for i in range(batch_size):
			result_vec[i] = matrix_sample.reshape(embedding_dim, n_distributions, batch_size)[:, current_dist_states_indices[i], i]
		return np.asarray(result_vec, dtype=np.float32)





class UniformSample(Sampler):
	def get_sample(self, batch_size, embedding_dim, n_distributions):
		return np.random.uniform(-1, 1, size=(batch_size, embedding_dim))


class MultiModalUniformSample(Sampler):
	def get_sample(self, batch_size, embedding_dim, n_distributions):
		current_dist_states_indices = np.random.randint(0, n_distributions - 1, batch_size)

		result_vec = np.zeros((batch_size, embedding_dim))
		for i in range(batch_size):
			result_vec[i] = np.random.uniform(-1 + current_dist_states_indices[i], 1 + current_dist_states_indices[i],size=embedding_dim)
		return np.asarray(result_vec, dtype=np.float32)


if __name__ == '__main__':
	import matplotlib.pyplot as plt

	test = MultiModalUniformSample()
	test_uni = UniformSample()
	test_mul_uni=MultiModalUniformSample()
	a = test.get_sample(10, 5, 10)
	b = test_uni.get_sample(10, 5, 10)
	c = test_mul_uni.get_sample(10, 5, 10)
	print(a)
	plt.plot(a)
	plt.show()
	plt.plot(b)
	plt.show()
	plt.plot(c)
	plt.show()
