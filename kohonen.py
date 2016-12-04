"""
Miniproject1: Unsupervised and Reinforcement Learning.
authors: Sharbatanu Chatterjee, Andras Ecker
"""

import numpy as np
import matplotlib.pyplot as plt
from helpers import *
from figures import *


def som_step(size_k, centers, data, neighbor, eta, sigma):
	"""
	Performs one step of the sequential learning for a self-organized map (SOM).
	:param size_k: size of the Kohonen map
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


def kohonen(size_k, sigma, eta, tmax, thres_iter, exc6=True, data_range=10, dim=2, n_clust=1, size_c=100, var_c=4, show_progress=True):
	"""
	Learning in Kohonen map (using create_data, plot_data and som_step)
	:param size_k: size of the Kohonen map 
	:param sigma: width of the neighborhood
	:param eta: learning rate
	:param tmax: max iteration
	:param exc6: logical flag (just to distinguish data and plotting style)
	:param data_range: 255 for digits (miniproject), see create_data in helpers.py (exc6)
	:param dim: 28*28 for digits (miniproject), see create_data in helpers.py (exc6)	
	:param n_clust, size_c, var_c: params of create_data in helpers.py (exc6)
	:param show_progress: dispaly figure in every iteration (show_progress=False, faster-just saves the final figure)
	"""
	
	if exc6:
		import matplotlib.pylab as plb
		plb.close('all')  # just to make sure	
		data = create_data(n_clust, data_range, var_c, size_c, dim)
	else:
		data, ideal_prototypes = load_data("Sharbatanu Chatterjee", dim)  # load in numbers (miniproject1)
    
	# initialise the centers randomly
	centers = np.random.rand(size_k**2, dim) * data_range
    
	# build a neighborhood matrix
	neighbor = np.arange(size_k**2).reshape((size_k, size_k))
    
	# set the random order in which the datapoints should be presented
	i_random = np.arange(tmax) % data.shape[0]
	np.random.shuffle(i_random)
	
	if exc6:
		handles = None # init handles
		plb.ion() # turn "interaction" on in pylab
	
	# MAEs
	MSEs = []	
	prev_centers = np.copy(centers)
	sigma_step = (sigma-0.2)/tmax # decrease sigma to 0.2 during the whole process

	###

	#possible online version, define arbitrarily that we'll stop if for a threshold of thres_iter iterations (thres_iter presentations of data vectors), the CV increases rather than decreases

	varMSEs = []
	meanMSEs = []
	CVMSEs = []

	# thres_iter = 5
	step_size = 10
	count = 0
	prev_step = 0

	###

	# core of the code:
	for it, i in enumerate(i_random):
		#centers = som_step(size_k, centers, data[i,:], neighbor, eta, sigma)  # update centers
		#change sigma to 1,3,5 and get the results at least
		centers = som_step(size_k, centers, data[i,:], neighbor, eta, sigma=(sigma-it*sigma_step))  # update centers, decreasing sigma

		#MAEs.append(np.sum(np.abs(prev_centers-centers)) / data.shape[0])

		MSEs.append(np.sum((prev_centers-centers)**2)/data.shape[0]) #the paper looks at MSE, so just in case

		prev_centers = np.copy(centers)	

		#### add online version here ####
		
		meanMSEs.append(np.mean(MSEs[0:it]))
		varMSEs.append(np.var(MSEs[0:it]))

		CV = 100*((np.sqrt(np.var(MSEs[0:it])))/(np.mean(MSEs[0:it])))

		CVMSEs.append(CV)

		for j in np.arange(max((it-thres_iter*step_size),step_size),max(it,2*step_size),step_size):
			curr_step = np.mean(CVMSEs[(j-step_size):j])
			count += int(curr_step > prev_step)
			prev_step = curr_step
			#print(count)

		if count == thres_iter:
			print(count, it)
			break
		else :
			count=0

		#####

		if exc6:
			handles = plot_data(centers, data, neighbor, sigma, eta, it, tmax, handles)  # update plot
			if show_progress: # for fancy iterative plot ! (set show=False to not show it)
				plb.draw()
				plb.show()
				plb.pause(1-7)
		
	
	if exc6:
		# leave the window open at the end of the loop for 3s and save fig
		plb.draw()
		plb.show()
		plb.waitforbuttonpress(timeout=3)
		# save figure
		from datetime import datetime as dt
		time_ = dt.now().strftime('%Y-%m-%d_%H:%M:%S')
		figName = 'figures/kohonen_it:%s(%s).png'%(tmax, time_)
		plb.savefig(figName)
	else:
		#plot_results(size_k, centers, ideal_prototypes, MAEs)
		plot_results(size_k, centers, ideal_prototypes, MSEs, eta, sigma, step_size)

	print("-----terminated; figure saved!-----")


if __name__ == "__main__":
	
	#size_k = [6,7] #Sharbat
	size_k = [8,10] #Andras
	
	# set the width of the neighborhood (via the width of the gaussian that describes it)
	sigma_factor = 6
	#sigma = np.sqrt(2.0 * size_k**2) / sigma_factor  # in the "real world" network size and the size of the neighbourhod should be correlated (exc6)
	sigma = [1,3,5]  # hovewer, in the project they ask us to be stupid and play around and hard code sigma (to 1,3,5 - miniproject)
	etas = np.arange(0.1,1,0.1)
	thres_iter = [3,4,5]


#kohonen(size_k=size_k, sigma=sigma, eta=0.3, tmax=400)  # for exc6
	for s in size_k:
		for sig in sigma:
			for rate in etas:
				for t in thres_iter:
					kohonen(size_k=s, sigma=sig, eta=rate, tmax=10000, thres_iter=t, exc6=False, data_range=255, dim=28*28)  # for miniproject


