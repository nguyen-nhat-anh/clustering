import numpy as np
from scipy.spatial.distance import pdist, squareform

def compute_dunn_index(X, cluster_labels, n_clusters, metric='correlation'):
    d = squareform(pdist(X, metric=metric))
    n_samples, n_features = X.shape

    min_distance = 100
    max_diam = 0
    for i in range(n_samples):
        for j in range(i+1,n_samples):
            if cluster_labels[i] != cluster_labels[j]:
                if d[i,j] < min_distance:
                    min_distance = d[i,j]
            else:
                if d[i,j] > max_diam:
                    max_diam = d[i,j]

    dunn_index = min_distance/max_diam
    print("For n_clusters =", n_clusters,
          "the dunn index is:", dunn_index)



