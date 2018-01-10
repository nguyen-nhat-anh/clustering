import numpy as np
import itertools
import clustering_algorithm
import matplotlib.pyplot as plt
from collections import Counter

def mbsas_threshold_plot(X, threshold_range=np.arange(0.5,1.5,0.05), max_n_cluster=10, metric='correlation', merge_threshold=0.1):
    n_clusters_range = []
    n_samples, n_features = X.shape
    n_loops=10
    a=0
    for threshold in threshold_range:
        a=a+1
        print('loop', a)
        n_clusters = []
        for i in range(n_loops):
            labels = clustering_algorithm.modified_bsas(np.random.permutation(X), threshold, max_n_clusters=max_n_cluster,
                                                        metric=metric, refine=True, merge_threshold=merge_threshold)
            n_clusters.append(labels.max()+1)
        most_common,_ = Counter(n_clusters).most_common(1)[0]
        n_clusters_range = np.append(n_clusters_range,most_common)
    plt.plot(threshold_range,n_clusters_range,'-b')
    plt.show()


