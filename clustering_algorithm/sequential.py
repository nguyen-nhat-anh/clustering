import numpy as np
from scipy.spatial.distance import pdist, squareform


def bsas(data, threshold, max_n_clusters, metric='correlation', refine=False, merge_threshold=None):
    d = squareform(pdist(data,metric=metric))
    n_samples, n_features = data.shape
    m = 0

    labels = np.zeros(shape=n_samples, dtype=int)
    labels[0] = m

    # distance from data point i to cluster j
    dist_to_cluster = np.zeros(shape=(n_samples, max_n_clusters), dtype=float)
    dist_to_cluster[:, m] = d[:, 0]

    for i in range(1, n_samples):
        min_index = dist_to_cluster[i, 0:(m+1)].argmin()
        if dist_to_cluster[i, min_index] > threshold and m + 1 < max_n_clusters:
            m += 1
            labels[i] = m
            dist_to_cluster[:, m] = d[:, i]
        else:
            labels[i] = min_index
            dist_to_cluster[:, min_index] *= labels[labels == min_index].size - 1
            dist_to_cluster[:, min_index] += d[:, i]
            dist_to_cluster[:, min_index] /= labels[labels == min_index].size

    if refine:
        labels = _merge_clusters(d, labels, merge_threshold)
        labels = _reassign_data(d, labels)

    return labels


def modified_bsas(data, threshold, max_n_clusters, metric='correlation', refine=False, merge_threshold=None):
    d = squareform(pdist(data,metric=metric))
    n_samples, n_features = data.shape
    m = 0

    labels = np.zeros(shape=n_samples, dtype=int)
    labels.fill(-1) # -1 means unassigned
    labels[0] = m

    # distance from data point i to cluster j
    dist_to_cluster = np.zeros(shape=(n_samples, max_n_clusters), dtype=float)
    dist_to_cluster[:, m] = d[:, 0]

    for i in range(1, n_samples):
        min_index = dist_to_cluster[i, 0:(m + 1)].argmin()
        if dist_to_cluster[i, min_index] > threshold and m + 1 < max_n_clusters:
            m += 1
            labels[i] = m
            dist_to_cluster[:,m] = d[:,i]

    for i in range(n_samples):
        if labels[i] == -1:
            min_index = dist_to_cluster[i, 0:(m + 1)].argmin()
            labels[i] = min_index
            dist_to_cluster[:,min_index] *= labels[labels == min_index].size - 1
            dist_to_cluster[:,min_index] += d[:,i]
            dist_to_cluster[:, min_index] /= labels[labels == min_index].size

    if refine:
        labels = _merge_clusters(d, labels, merge_threshold)
        labels = _reassign_data(d, labels)

    return labels


def _merge_clusters(distance_matrix, labels, merge_threshold):
    m = labels.max() + 1
    i = 0
    while i < m - 1:
        flag = False
        while flag == False:
            d = np.zeros(shape=m, dtype=float)
            for j in range(i+1, m):
                d[j] = np.min(distance_matrix[np.ix_(labels == i, labels == j)])
            min_index = i + 1 + d[i+1:m].argmin()
            if d[min_index] <= merge_threshold:
                labels[labels == min_index] = i
                for j in range(min_index+1, m):
                    labels[labels == j] = j-1
                m -= 1
                if i == m - 1:
                    flag = True
            else:
                i += 1
                flag = True

    return labels


def _reassign_data(distance_matrix, labels):
    n_samples = labels.shape[0]
    n_cluster = labels.max() + 1
    b = np.zeros(shape=n_samples,dtype=int)
    for i in range(n_samples):
        d = np.zeros(shape=n_cluster,dtype=float)
        for j in range(n_cluster):
            d[j] = distance_matrix[i, labels==j].mean()
        b[i] = d.argmin()

    return b
