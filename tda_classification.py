import numpy as np;
import sklearn;
from sklearn import datasets;
from sklearn.linear_model import LogisticRegression;
from sklearn.model_selection import train_test_split;

import matplotlib.pyplot as plt;

from ripser import Rips;
from persim import PersImage;
from persim import PersistenceImager;



#
# Construct two datasets
# 	• noise data
# 	• circle data
#

n = 200;
n_per_class = int(n/2);
n_in_class = 2*n;



def noise (n, scale):
	return scale * np.random.random((n,2));

def circle(n, scale, offset):
	return offset + scale*datasets.make_circles(n_samples=n, factor=.40, noise=.05)[0];



noise_only = [noise(n_in_class, 150) for _ in range(n_per_class)];



half = int(n_in_class / 2);

with_circle = [np.concatenate((circle(half, 50, 70), noise(half, 150))) for _ in range(n_per_class)];





datas = [];
datas.extend(noise_only);
datas.extend(with_circle);


labels = np.zeros(n);
labels[n_per_class:] = 1;




#
# Visualize data
#

fig, axs = plt.subplots(1,2);
fig.set_size_inches (10, 5);

xs = noise_only[0][:,0];
ys = noise_only[0][:,1];

axs[0].scatter(xs, ys);
axs[0].set_title('Noise');
axs[0].set_aspect('equal', 'box');


xs1 = with_circle[0][:,0];
ys1 = with_circle[0][:,1];

axs[1].scatter(xs1, ys1);
axs[1].set_title('Circlular');
axs[1].set_aspect('equal', 'box');




#
# Compute homology of the circular and noise dataset
#

rips = Rips(maxdim=1, coeff=2);

diagrams = [rips.fit_transform(data) for data in datas];
diagrams_h1 = [rips.fit_transform(data) for data in datas];


#
# Generate persistence diagrams of H1 for circluar/noise dataset
#

plt.figure(figsize=(12,6));
plt.subplot(121);

rips.plot(diagrams_h1[0], show=False);
plt.title('Noise Data Persistence Diagram for H1');

plt.subplot(122);
rips.plot(diagrams_h1[-1], show=False);
plt.title('Circle Persistence Diagram for H1');

# plt.show();



#
# Convert persistence diagrams into persistence images
#

# p_imgr = PersistenceImager(pixel_size=1);
# p_imgr.fit(diagrams_h1);

# imgs = p_imgr.transform(diagrams_h1);





# xs, ys = noise_only[0][:0], noise_only[0][:1];
# axs[0].scatter(xs,ys);
# axs[0].set_title('Noise');
# axs[0].set_aspect('equal', 'box');

# xs1, ys1 = with_circle[0][:0], with_circle[0][:1];

# axs[1].scatter(xs1, ys1);
# axs[1].set_title('Circle with Noise');
# axs[1].set_aspect('equal', 'box');

# fig.tight_layout();






























