import numpy as np;
import matplotlib.pyplot as plt;
import matplotlib.gridspec as gridspec;
import networkx as nx;
from IPython.display import Video;

import ripser;
import persim;

import teaspoon.MakeData.PointCloud as makePtCloud;
import teaspoon.TDA.Draw as Draw;

from teaspoon.SP.network import ordinal_partition_graph;
from teaspoon.TDA.PHN import PH_network;
from teaspoon.SP.network_tools import make_network;
from teaspoon.parameter_selection.MsPE import MsPE_tau;

import teaspoon.MakeData.DynSysLib.DynSysLib as DSL;


print('hello, world!');





#
# Generate Annulus shaped point cloud for testing
#

r = 1;
R = 2;

P = makePtCloud.Annulus(N=200, r=r, R=R, seed=None);

plt.scatter(P[:,0], P[:,1]);



#
# Function to conveniently plot diagrams
#

def drawTDA (P, diagrams, R=2):
	fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20,5));

	# Draw point cloud
	plt.sca(axes[0]);
	plt.title('Point Cloud');
	plt.scatter(P[:,0], P[:,1]);

	# Draw diagrams
	plt.sca(axes[1]);
	plt.title('0-Dim Diagram');
	Draw.drawDgm(diagrams[0]);

	plt.sca(axes[2]);
	plt.title('1-Dim Diagram');
	Draw.drawDgm(diagrams[1]);

	plt.axis([0,R,0,R]);



#
# Example - Point Cloud, 0-Dimensional and 1-Dimensional Persistence Diagrams for Random Noise
#

# P = makePtCloud.Cube();
# diagrams = ripser.ripser(P)['dgms'];

# drawTDA (P=P, diagrams=diagrams, R=.80);

drawTDA(P=makePtCloud.Cube(), diagrams=ripser.ripser(makePtCloud.Cube())['dgms'], R=.80);



#
# Example - Double Annulus
#

def DoubleAnnulus (r1=1, R1=1, r2=.80, R2=1.30, xshift=1.3):
	P = makePtCloud.Annulus(r=r1, R=R1);
	Q = makePtCloud.Annulus(r=r2, R=R2);

	Q[:,0] = Q[:,0] + xshift;
	P = np.concatenate((P,Q));

	return P;


P = DoubleAnnulus(r1=1, R1=2, r2=.5, R2=1.30, xshift=3);
plt.scatter(P[:,0], P[:,1]);




#
# Simple example by-hand
#

D1 = np.array([
	[0, 1, np.inf, np.inf, 6],
	[0, 0, 5, np.inf, np.inf],
	[0, 0, 0, 2, 4],
	[0, 0, 0, 0, 3],
	[0, 0, 0, 0, 0]
]);

print(D1);
print(D1.shape);

D = D1 + D1.T;

# Define diagram with distance matrix instead of point cloud
diagrams = ripser.ripser(D, distance_matrix = True, maxdim=1)['dgms'];

print('0-Dim Diagram'); print(diagrams[0]);
print('1-Dim Diagram'); print(diagrams[1]);




#
# - Examples
#



#
# Function to draw graph
#

def drawGraphEx (G):
	pos = nx.spring_layout(G);
	nx.draw_networkx_nodes (G, pos, node_size=70); # draw nodes
	nx.draw_networkx_edges (G, pos, width=2);

	edge_labels = nx.draw_networkx_edge_labels (G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'));




#
# Create Erdos-Renyii Random Graph
#

n=10; p=.30;

G = nx.erdos_renyi_graph (n, p, seed=None, directed=False);

m = len(G.edges);

print('there are {} edges'.format(m));

max_weight = 100;
weights = np.random.randint (max_weight, size=m);

for i,e in enumerate(G.edges()):
	G[e[0]][e[1]]['weight'] = weights[i];


#
# Plot the graph
#

nx.draw(G); # plt.show();



# Add weights to adjacency matrix

A = nx.adjacency_matrix(G, weight='weight');
A = A.todense();
A = np.array(A);
A = A.astype('float64');
A[np.where(A==0)] = np.inf;
np.fill_diagonal(A,0);

plt.colorbar(plt.matshow(A,vmax=100));

# plt.show();















































