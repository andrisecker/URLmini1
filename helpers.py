"""
helper file for exc6 and miniproject1
"""

import numpy as np
from figures import plot_ideal_prototypes


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


def name2digits(name):
	"""
	takes your name and converts it into a pseudo-random selection of 4 digits from 0-9.
	:param name: (string) your name! whao:D
	:return list of 4 selected digits
	"""
   
	name = name.lower()   
	if len(name)>25:
		name = name[0:25]
        
	primenumbers = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97]   
	n = len(name)    
	s = 0.0
    
	for i in range(n):
		s += primenumbers[i] * ord(name[i])*2.0**(i+1)

	import scipy.io.matlab
	Data = scipy.io.matlab.loadmat('hash.mat', struct_as_record=True)
	x = Data['x']
	t = np.mod(s, x.shape[0])

	return np.sort(x[t, :])


def load_data(name, dim):
	"""
	loads in data (and labels), selects 4 digits to work with and creates ideal prototypes (by averaging the prototypes corresponding to one digit)
	:param name: (string) used by name2digits
	:param dim: used only to create labels
	:return red_data: data matrix used in the miniproject
	:return ideal_prototypes: (dictionary) key: label, val: coordinates of the prototype (1*784)
	"""

	# load in data and labels
	data = np.array(np.loadtxt('data/data.txt'))
	labels = np.loadtxt('data/labels.txt')

	targetdigits = name2digits(name)

	red_data = data[np.logical_or.reduce([labels==x for x in targetdigits]),:]

	# assign labels to the results
	ideal_prototypes = {}
	for td in targetdigits:
		inds = np.where(labels == td)[0]
		ideal_prototypes[td] = np.mean(data[inds,:], axis=0)

	plot_ideal_prototypes(ideal_prototypes)

	return red_data, ideal_prototypes




