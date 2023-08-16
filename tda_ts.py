import numpy as np;
import matplotlib as plt;
from gudhi import plot_diagram, plot_persistence_barcode;

import ripser;


#
# Generate time-series temperature data
#

np.random.seed(0);
num_points = 100;
time = np.linspace(0, 10, num_points);
temp = np.sin(time) + np.random.normal(0, .20, num_points);

#
# Create distance matrix
#

distance_mat = np.abs(np.subtract.outer(temp, temp));

#
# Persistent Homology Computations
#
# 	• compute persistent homology of the distance matrix
# 	• obtain topological information about data (connected components, loops, voids)
#

rip = ripser(distance_mat, maxdim=1);

#
# Plot persistence diagrams
#

plot_diagram(rip['dgms']);
plt.title('Persistence Diagram');
# plt.show();


#
# Plot persistence barcode
#

plot_persistence_barcode(rip['dgms']);
plt.title('Persistence Barcode');
# plt.show();
 




  




  




  




  




  




  




  




  




  




  




  




  




