"""
Miniproject1: Unsupervised and Reinforcement Learning.
authors: Sharbatanu Chatterjee, Andras Ecker
"""

import numpy as np
import matplotlib.pylab as plb
import matplotlib.pyplot as plt
from helpers import *
from figures import *


def som_step(size_k, centers, data, neighbor, eta, sigma):
	"""
	Performs one step of the sequential learning for a self-organized map (SOM).
	:param size_k:
    :param centers:  matrix - cluster centres. (center x dimension) (in this case dim=2)
	:param data: vector - the actually presented datapoint to be presented in this timestep
	:param neighbor: matrix - the coordinates of the centers in the desired neighborhood.
	:param eta: learning rate
    :param sigma: the width of the gaussian neighborhood function. (Effectively describing the width of the neighborhood)
	:return updated centers
    """

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


def kohonen(size_k, sigma, eta, tmax, data_range=10, dim=2, show_progress=True, n_clust=1, size_c=100, var_c=4):
	"""
	Learning in Kohonen map (using create_data, plot_data and som_step)
	:param size_k: size of the Kohonen map 
	:param sigma: width of the neighborhood
	:param eta: learning rate
	:param tmax: max iteration  #TODO make a better convergence crit!
	:param data_range: 255 for digits (miniproject), see create_data in helpers.py (exc6)
	:param dim: 28*28 for digits (miniproject), see create_data in helpers.py (exc6)
	:param show_progress: dispaly figure in every iteration (show_progress=False, faster-just saves the final figure)
	:param n_clust, size_c, var_c: params of create_data in helpers.py (exc6)
	"""

	plb.close('all')
	
	#data = create_data(n_clust, data_range, var_c, size_c, dim) # create data (exc6)
	data = load_data("Andras Ecker")  # load in numbers (miniproject1)
	dy, dx = data.shape
    
	# initialise the centers randomly
	centers = np.random.rand(size_k**2, dim) * data_range
    
	# build a neighborhood matrix
	neighbor = np.arange(size_k**2).reshape((size_k, size_k))
    
	# set the random order in which the datapoints should be presented
	i_random = np.arange(tmax) % dy
	np.random.shuffle(i_random)
		
	handles = None # init handles
	if show_progress: # for fancy iterative plot ! (set show=False to not show it)
		plb.ion() # turn "interaction" on in pylab	
	

	# core of the code:
	for it, i in enumerate(i_random):
		centers = som_step(size_k, centers, data[i,:], neighbor, eta, sigma)  # update centers
		#handles = plot_data(centers, data, neighbor, sigma, eta, it, tmax, handles)  # update plot
		if show_progress:	
			plb.draw()
			plb.show()
			plb.pause(1e-7)
		
	
	if show_progress:
	# leave the window open at the end of the loop
		plb.draw()
		plb.show()
		plb.waitforbuttonpress(timeout=3)

		# save figure
		from datetime import datetime as dt
		time_ = dt.now().strftime('%Y-%m-%d_%H:%M:%S')
		figName = 'figures/kohonen_it:%s(%s).jpg'%(tmax, time_)
		plb.savefig(figName)

	print("-----terminated; figure saved!-----")


	# for visualization, you can use this:
	fig = plt.figure(figsize=(10,8))
	for i in range(size_k**2):
		ax = fig.add_subplot(size_k, size_k, i+1)
        
		ax.imshow(np.reshape(centers[i,:], [28, 28]),interpolation='bilinear')
		#plt.axis('off')

	plt.show()


if __name__ == "__main__":
	
	size_k = 6
	
	# set the width of the neighborhood (via the width of the gaussian that describes it)
	sigma_factor = 6
	#sigma = np.sqrt(2.0 * size_k**2) / sigma_factor  # in the "real world" network size and the size of the neighbourhod should be correlated (exc6)
	sigma = 3  # hovewer, in the project they ask us to be stupid and play around and hard code sigma (to 1,3,5 - miniproject)

	#kohonen(size_k=size_k, sigma=sigma, eta=0.3, tmax=400)  # for exc6
	kohonen(size_k=size_k, sigma=sigma, eta=0.3, tmax=1000, data_range=255, dim=28*28, show_progress=False)


