import numpy as np
import ripser
import matplotlib.pyplot as plt
import random
from persim.persistent_entropy import *
from scipy import stats




#
# Construct normal data
#

mu = .50; sigma = .250;
l1 = [];

for i in range(10):
	# d1 = np.random.random(mu, sigma, (50,2));
	d1 = np.random.normal(mu, sigma, (50,2));
	l1.append(d1);

#
# Construct uniform point cloud
#

l2 = [];
for i in range(10):
	d2 = np.random.random((50,2));
	l2.append(d2);



#
# Plot the normal and unfirom point clouds
#

plt.scatter(d1[:,0], d1[:,1], label='Normal');
plt.scatter(d2[:,0], d2[:,1], label='Uniform');

plt.axis('equal');
plt.legend();

# plt.show();




#
# Generate persistence diagrams for normal/uniform distributions
#

p = 0;

diagrams_d1 = [];
diagrams_d2 = [];

for i in range(len(l1)):
	diagrams_d1.append(ripser.ripser(l1[i])['dgms'][p]);
	diagrams_d2.append(ripser.ripser(l2[i])['dgms'][p]);


#
# Compute persistent entropy for each persistence diagram
#

entropy_d1 = persistent_entropy(diagrams_d1);
entropy_d2 = persistent_entropy(diagrams_d2);


#
# Perform Mann-Whitney Test
# 	• H0: geometry of the point clouds is the same
# 	• Ha: point cloud geometries are different
#

np.random.seed(10);

stats.mannwhitneyu(entropy_d1, entropy_d2);
# MannwhitneyuResult(statistic=13.0, pvalue=0.00579535854433471)































