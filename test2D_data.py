import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from clustering_algorithm.k_means import k_means
from clustering_algorithm.sequential import *
#from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal

data = []
dist = multivariate_normal(mean=[0,0], cov=[[1,0],[0,1]])
for i in range(150):
    data.append(dist.rvs())
dist = multivariate_normal(mean=[5,5], cov=[[1,0],[0,1]])
for i in range(150):
    data.append(dist.rvs())
dist = multivariate_normal(mean=[10, 1], cov=[[1, 0], [0, 1]])
for i in range(150):
    data.append(dist.rvs())

data = np.asarray(data)
#plt.scatter(data[:,0],data[:,1])
#plt.show()

#estimator = KMeans(n_clusters=3)
#estimator.fit(data)
#labels = estimator.labels_

# _, labels = k_means(data, n_clusters=3, algorithm='random')
# print(labels)

labels = modified_bsas(data, 4, 7,refine=True, merge_threshold=0.2)

def set_colors(labels, colors='rgbykcm'):
    colored_labels = []
    for label in labels:
        colored_labels.append(colors[label])
    return colored_labels

colors = set_colors(labels)
plt.scatter(data[:,0], data[:,1], c=colors)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

labels = modified_bsas(data, 4, 7, refine=True, merge_threshold=0.2, reassign=True)

colors = set_colors(labels)
plt.scatter(data[:,0], data[:,1], c=colors)
plt.xlabel("x")
plt.ylabel("y")
plt.show()