"""
helper file for plots
#TODO: make fancy figure for changing the eta, sigma, size 
"""

import numpy as np
import matplotlib.pylab as plb
import matplotlib.pyplot as plt


def plot_data(centers, data, neighbor, sigma, eta, it, tmax, handles):
	"""Plot self-organising map (SOM) data (exc6)
	This includes the used datapoints as well as the cluster centres and neighborhoods.
 	:param centers:  (matrix) cluster centres to be plotted. Have to be in format: center X dimension (in this case 2)
	:param data: (matrix) datapoints to be plotted. have to be in the same format as centers
    :param neighbor: (matrix) the coordinates of the centers in the desired neighborhood.
	:param sigma, eta, it, tmax: just for nice title ...
	:return updated handles
	"""

	if handles is None:  # it is the first time we call this method: create the graphs
		a, b = centers.shape
        
		# handles = [figure, scatter_centers, scatter_data, plot_centers_1, plot_centers_2, ...]
		handles = []
		handles.append(plb.figure(figsize=(10,8)))
        
		# plot centers and datapoints
		handles.append(plb.scatter(centers[:, 0], centers[:, 1], c='r', marker='x', facecolor="w", edgecolor='r', label="centers"))
		handles.append(plb.scatter(data[:, 0],data[:, 1],c = 'b', marker='o', facecolor="w", edgecolor='b', label="data"))
        
		# plot neighborhood grid
		h1 = []; h2 = []; handles.append(h1); handles.append(h2)
		for g in range(len(neighbor)):
			h1.append(plb.plot(centers[neighbor[g, :],0], centers[neighbor[g, :],1], 'k')[0])
		for g in range(len(neighbor)):
			h2.append(plb.plot(centers[neighbor[:, g],0], centers[neighbor[:, g],1], 'k')[0])

	else:  # the graphs already exist, we have to plot the data

		# update centers
		handles[1].set_offsets(centers)
		# update paths
		for g in range(len(neighbor)):
			handles[3][g].set_xdata(centers[neighbor[g, :],0])
			handles[3][g].set_ydata(centers[neighbor[g, :],1])
			handles[4][g].set_xdata(centers[neighbor[:, g],0])
			handles[4][g].set_ydata(centers[neighbor[:, g],1])

	handles.append(plb.title("SOM; sigma:%.4f, eta:%s, it:%s/%s"%(sigma, eta, it+1, tmax)))
	handles.append(plb.xlabel("x"))
	handles.append(plb.ylabel("y"))
	handles.append(plb.legend())
    
	# take care of the zoom
	xmax = np.max(data[:,0] + 1)
	xmin = np.min(data[:,0] - 1)
	ymax = np.max(data[:,1] + 1)
	ymin = np.min(data[:,1] - 1)
	xmax = max(xmax, np.max(centers[:,0] + 0.5))
	xmin = min(xmin, np.min(centers[:,0] - 0.5))
	ymax = max(ymax, np.max(centers[:,1] + 0.5))
	ymin = min(ymin, np.min(centers[:,1] - 0.5))

	plb.axis(xmin = xmin, xmax = xmax, ymin=ymin, ymax=ymax)

	return handles

# ===========================================================================================================


def plot_results(size_k, centers):
	"""
	plots the predicted digites (and saves the plot)
	:param size_k: size of the Kohonen map
	:param centers: (matrix) cluster centres to be plotted
	"""

	fig = plt.figure(figsize=(10,8))
	for i in range(size_k**2):
		ax = fig.add_subplot(size_k, size_k, i+1)
        
		ax.imshow(np.reshape(centers[i,:], [28, 28]),interpolation='bilinear')
		plt.axis('off')

	# save figure
	from datetime import datetime as dt
	time_ = dt.now().strftime('%Y-%m-%d_%H:%M:%S')
	figName = 'figures/kohonen_(%s).jpg'%(time_)
	fig.savefig(figName)

	plt.show()

