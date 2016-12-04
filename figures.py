"""
helper file for plots
#TODO: make fancy figure for changing the eta, sigma, size 
"""

import numpy as np
import matplotlib.pylab as plb
import matplotlib.pyplot as plt
from datetime import datetime as dt

mplPars = { #'text.usetex'       :    True,
            'axes.labelsize'    :   'large',
            'font.family'       :   'serif',
            'font.sans-serif'   :   'computer modern roman',
            'font.size'         :    18,
            'xtick.labelsize'   :    16,
            'ytick.labelsize'   :    16
            }
for key, val in mplPars.items():
    plt.rcParams[key] = val


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


def plot_ideal_prototypes(ideal_prototypes):
	"""
	plots the ideal prototypes (made by averaging the prototypes corresponding to one digit)
	:param ideal_prototypes: (dictionary) key: label, val: coordinates of the prototype (1*784)
	"""

	fig = plt.figure(figsize=(10,8))
	
	
	for i, (label, val) in enumerate(ideal_prototypes.iteritems(), 1):  # look at this loop !
		ax = fig.add_subplot(2, 2, i)		
        
		ax.imshow(np.reshape(val, [28, 28]),interpolation='bilinear')
		ax.set_title(label)
		plt.axis('off')

	#fig.tight_layout()
	figName = 'figures/ideal_prototypes.jpg'
	fig.savefig(figName)
	#plt.show()

def plot_errors(MAEs, step_size=200):
	"""
	plots "convergence criteria"
	:param MAEs: (list) Mean Absolute Erros (prev iteration, current iteration)
	:param step_size: step size for computing mean and var
	"""

	points = np.arange(step_size, len(MAEs), step_size)

	meanMAEs = []
	varMAEs = []
	CVMAEs = []
	for p in points:
		meanMAEs.append(np.mean(MAEs[0:p]))
		varMAEs.append(np.var(MAEs[0:p]))
		CV = 100*((np.sqrt(np.var(MAEs[0:p])))/(np.mean(MAEs[0:p])))
		CVMAEs.append(CV)

	
	fig = plt.figure(figsize=(10,8))
	ax = fig.add_subplot(3,1,1)

	ax.plot(MAEs, 'b-', marker='|', linewidth=2, label="MAE")
	ax.set_title("Mean Absolute Error")
	ax.set_xlabel("iteration")
	ax.set_xlim([0, len(MAEs)-1])
	ax.legend()

	ax2 = fig.add_subplot(3,1,2)
	ax3 = ax2.twinx()

	ax3.plot(meanMAEs, 'r-', marker='|', linewidth=2, label="mean(MAE)")
	ax3.set_ylabel(ylabel='mean(MAE)', color='red')
	ax2.plot(varMAEs, 'b-', marker='|', linewidth=2, label="var(MAE)")
	ax2.set_ylabel(ylabel='var(MAE)', color='blue')
	ax2.set_title("mean and var MAE")
	ax2.set_xlabel("*%s iteration"%step_size)
	ax2.set_xlim([0, len(meanMAEs)-1])
	ax2.legend(loc=2)
	ax3.legend(loc=1)

	ax4 = fig.add_subplot(3,1,3)
	ax4.plot(CVMAEs, 'r-', marker='|', linewidth=2, label="CV")
	ax4.set_ylabel(ylabel='CV', color='red')

	fig.tight_layout()

	# save figure	
	time_ = dt.now().strftime('%Y-%m-%d_%H:%M:%S')
	figName = 'figures/kohonen_MAE_(%s).png'%(time_)
	fig.savefig(figName)
	
	#plt.show()

def plot_results(size_k, centers, ideal_prototypes, MAEs):
	"""
	plots the predicted digites (and saves the plot)
	:param size_k: size of the Kohonen map
	:param centers: (matrix) cluster centres to be plotted
	:param ideal_prototypes: (map) with ideal prototypes (average of every sample belonging, to that prototype) -> used to assign labels
	:param MASs: used by plot_errors
	"""

	fig = plt.figure(figsize=(10,8))
	for i in range(size_k**2):
		
		# calculate closest prototype (identified with labels)
		tmp = {}
		for key, val in ideal_prototypes.items():
			tmp[key] = np.sum((centers[i,:] - val)**2) / centers.shape[1]
		#print(tmp)
		label = min(tmp, key=tmp.get)
		
		ax = fig.add_subplot(size_k, size_k, i+1)
        
		ax.imshow(np.reshape(centers[i,:], [28, 28]),interpolation='bilinear')
		#print("min:%s, max:%s"%(np.min(centers[i,:]), np.min(centers[i,:])))
		ax.set_title(label)
		plt.axis('off')

	#fig.tight_layout()

	# save figure	
	time_ = dt.now().strftime('%Y-%m-%d_%H:%M:%S')
	figName = 'figures/kohonen_(%s).png'%(time_)
	fig.savefig(figName)

	plot_errors(MAEs)

	plt.show()

