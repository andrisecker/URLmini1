"""
Miniproject1: Unsupervised and Reinforcement Learning.
authors: Sharbatanu Chatterjee, András Ecker
"""

import numpy as np
import matplotlib.pylab as plb
from helpers import *
from figures import *


def som_step(centers, data, neighbor, eta, sigma):
	"""
	Performs one step of the sequential learning for a self-organized map (SOM).
    :param centers:  matrix - cluster centres. (center x dimension) (in this case dim=2)
	:param data: vector - the actually presented datapoint to be presented in this timestep
	:param neighbor: matrix - the coordinates of the centers in the desired neighborhood.
	:param eta: learning rate
    :param sigma: the width of the gaussian neighborhood function. (Effectively describing the width of the neighborhood)
	:return updated centers
    """
    
	size_k = int(np.sqrt(len(centers)))
    
	# find the best matching unit via the minimal distance to the datapoint
	b = np.argmin(np.sum((centers - np.resize(data, (size_k**2, data.size)))**2, 1))
	# update winner
	centers[b, :] += eta * (data - centers[b, :])

	a, b = np.nonzero(neighbor == b)
	for j in range(size_k**2):
		a1, b1 = np.nonzero(neighbor == j)
		if j != b:
			# compute the discount to the update via the neighborhood function        
			disc = gauss(np.sqrt((a-a1)**2 + (b-b1)**2), [0, sigma])
		else:
			disc = 0

		# update non winners according to the neighborhood function    
		centers[j, :] += disc * eta * (data - centers[j,:])

	return centers


def kohonen(size_k, sigma, eta, tmax, show=True, n_clust=1, size_c=100, data_range=10, var_c=4, dim=2):
	"""
	Example for using create_data, plot_data and som_step
	:param size_k: size of the Kohonen map 
	:param sigma: width of the neighborhood
	:param eta: learning rate
	:param tmax: max iteration
	:param show: dispaly figure in every iteration (show=False, faster-just saves the final figure)
	:param n_clust, size_c, data_range, var_c, dim: params of create_data -> see there (helpers.py)
	"""

	plb.close('all')

	# create data
	data = create_data(n_clust, data_range, var_c, size_c, dim)
	dy, dx = data.shape
    
	# initialise the centers randomly
	centers = np.random.rand(size_k**2, dim) * data_range
    
	# build a neighborhood matrix
	neighbor = np.arange(size_k**2).reshape((size_k, size_k))
    
	# set the random order in which the datapoints should be presented
	i_random = np.arange(tmax) % dy
	np.random.shuffle(i_random)
	
	plb.ion() # turn "interaction" on in pylab	
	handles = None # init handles

	for it, i in enumerate(i_random):
		centers = som_step(centers, data[i,:], neighbor, eta, sigma)  # update centers
		handles = plot_data(centers, data, neighbor, sigma, eta, it, tmax, handles)  # update plot
		if show:
			plb.draw()
			plb.show()
			plb.pause(1e-7)
		
	# leave the window open at the end of the loop
	plb.draw()
	plb.show()
	plb.waitforbuttonpress(timeout=3)

	# save figures	
	from datetime import datetime as dt
	time_ = dt.now().strftime('%Y-%m-%d_%H:%M:%S')
	figName = 'figures/kohonen_it:%s(%s).jpg'%(tmax, time_)
	plb.savefig(figName)

	print("-----terminated; figure saved!-----")
    


if __name__ == "__main__":

	# set the width of the neighborhood (via the width of the gaussian that describes it)
	size_k = 6
	sigma = np.sqrt(2.0 * size_k**2)/6.0  #TODO check where 6 comes from... (maybe size_k?)

	kohonen(size_k=size_k, sigma=sigma, eta=0.3, tmax=400)


