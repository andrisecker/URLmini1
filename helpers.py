"""
helper file for miniproject1
"""

import numpy as np


def create_data(n_clust, data_range, var_c, size_c, dim):
	"""
	Create random data in form of clusters.
	:param n_clust: desired cluster count
	:param data_range: a rough estimation of the maximal values of the random data
	:param var_c: describes how big the cluster is allowed to get
	:param size_c: describes how many datapoints there are per cluster
	:param dim: dimensionality of the datapoints
    :return data matrix - (samples x dimension) (in this case dim=2)
	"""

	data = np.zeros((size_c*n_clust, dim))
	c = 0
	for i in range(n_clust):
		C = np.random.rand(1, dim) * data_range
		for j in range(size_c):
			data[c, :] = C + np.random.rand(1, dim) * (var_c/2) - var_c
			c += 1

	return data


def gauss(x,p):
	"""
	Return the gauss function N(x), with mean p[0] and std p[1].
	"""
	return np.exp((-(x - p[0])**2) / (2 * p[1]**2)) / np.sqrt(2*np.pi) /p[1]

# ===========================================================================================================
