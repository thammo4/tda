import numpy as np;
import matplotlib.pyplot as plt;

import ripser;
import persim;
import sklearn;
from sklearn import datasets;
from sklearn.metrics import pairwise_distances;



#
# Load iris dataset from sklearn
#

iris = datasets.load_iris();
d = iris.data;
target = iris.target;



#
# Construct distance matrix
#

# distance_matrix = np.sqrt(np.sum((data[:,np.newaxis]-data[np.newaxis,:])**2, axis=1));


# distance_matrix = np.sqrt(
# 	np.sum((data[:, np.newaxis] - data[np.newaxis, :]) ** 2, axis=-1)
# )

# distance_matrix = np.sqrt(np.sum((d[np.newaxis,:] - d[:, np.newaxis])**2, axis=1));

distance_matrix = pairwise_distances(d, metric='euclidean');

#
# Compute persistent homology
#

rips = ripser.ripser(X=distance_matrix, maxdim=2, distance_matrix=True, metric='euclidean');

#
# Construct Persistence Diagram
#

persistence_diagrams = rips['dgms'][1];

births = [pt[0] for pt in persistence_diagrams if np.isfinite(pt[1])];
deaths = [pt[1] for pt in persistence_diagrams if np.isfinite(pt[1])];

plt.scatter(births, deaths);

plt.xlabel('Birth Time'); plt.ylabel('Death Time');

plt.title('Iris Data Persistence Diagrams');

# plt.show();

























































































































































































































































































