import numpy as np
from scipy.spatial.distance import pdist, squareform


def bsas(data, threshold, max_n_clusters, metric='correlation', refine=False, merge_threshold=None):
    """ Clustering using Basic Sequential Algorithmic Scheme (BSAS)

    :param: data:numpy.array \n
        input data, array with shape (n_samples, n_features)
    :param: threshold:float \n
        threshold of dissimilarity
    :param: max_n_cluster:int \n
        maximum allowable number of clusters
    :param: metric:string,optional,default='correlation' \n
        dissimilarity measure
    :param: refine:boolean,optional,default=False \n
        choose whether or not to use refinement procedures (merging procedure and reassignment procedure)
    :param: merge_threshold:float,optional \n
        threshold of dissimilarity to merge two clusters into a single one in merging procedure
    :return: labels:numpy.array \n
        array of labels corresponding to input data, shape (n_samples,)
    """
    d = squareform(pdist(data,metric=metric)) # dissimilarity (distance) matrix of input data
    n_samples, n_features = data.shape # number of samples, number of features
    m = 0 # current number of clusters created

    labels = np.zeros(shape=n_samples, dtype=int)
    labels[0] = m

    # distance from data point i to cluster j
    dist_to_cluster = np.zeros(shape=(n_samples, max_n_clusters), dtype=float)
    dist_to_cluster[:, m] = d[:, 0]

    for i in range(1, n_samples):
        # find k where C_k is the nearest cluster of data point i
        min_index = dist_to_cluster[i, 0:(m+1)].argmin()
        if dist_to_cluster[i, min_index] > threshold and m + 1 < max_n_clusters:
            # create a new cluster and assign point i to it
            m += 1
            labels[i] = m
            dist_to_cluster[:, m] = d[:, i]
        else:
            # assign point i to nearest cluster C_k and update C_k distance information
            labels[i] = min_index
            dist_to_cluster[:, min_index] *= labels[labels == min_index].size - 1
            dist_to_cluster[:, min_index] += d[:, i]
            dist_to_cluster[:, min_index] /= labels[labels == min_index].size

    # Refinement stages
    if refine:
        labels = _merge_clusters(d, labels, merge_threshold)
        labels = _reassign_data(d, labels)

    return labels


def modified_bsas(data, threshold, max_n_clusters, metric='correlation', refine=False, merge_threshold=None):
    """ Clustering using Modified BSAS (MBSAS)

    :param: data:numpy.array \n
        input data, array with shape (n_samples, n_features)
    :param: threshold:float \n
        threshold of dissimilarity
    :param: max_n_cluster:int \n
        maximum allowable number of clusters
    :param: metric:string,optional,default='correlation' \n
        dissimilarity measure
    :param: refine:boolean,optional,default=False \n
        choose whether or not to use refinement procedures (merging procedure and reassignment procedure)
    :param: merge_threshold:float,optional \n
        threshold of dissimilarity to merge two clusters into a single one in merging procedure
    :return: labels:numpy.array \n
        array of labels corresponding to input data, shape (n_samples,)
    """
    d = squareform(pdist(data,metric=metric)) # dissimilarity (distance) matrix of input data
    n_samples, n_features = data.shape # number of samples, number of features
    m = 0 # current number of clusters created

    labels = np.zeros(shape=n_samples, dtype=int)
    labels.fill(-1) # -1 means unassigned
    labels[0] = m

    # distance from data point i to cluster j
    dist_to_cluster = np.zeros(shape=(n_samples, max_n_clusters), dtype=float)
    dist_to_cluster[:, m] = d[:, 0]

    # Phase 1: Cluster Determination
    for i in range(1, n_samples):
        min_index = dist_to_cluster[i, 0:(m + 1)].argmin()
        if dist_to_cluster[i, min_index] > threshold and m + 1 < max_n_clusters:
            m += 1
            labels[i] = m
            dist_to_cluster[:,m] = d[:,i]

    # Phase 2: Pattern Classification
    for i in range(n_samples):
        if labels[i] == -1:
            min_index = dist_to_cluster[i, 0:(m + 1)].argmin()
            labels[i] = min_index
            dist_to_cluster[:,min_index] *= labels[labels == min_index].size - 1
            dist_to_cluster[:,min_index] += d[:,i]
            dist_to_cluster[:, min_index] /= labels[labels == min_index].size

    # Refinement stages
    if refine:
        labels = _merge_clusters(d, labels, merge_threshold)
        labels = _reassign_data(d, labels)

    return labels


def _merge_clusters(distance_matrix, labels, merge_threshold):
    """ Merging procedure in refinement stage.

        :param: distance_matrix:numpy.array \n
            dissimilarity matrix of input data, array with shape (n_samples, n_sample)
        :param: labels:numpy.array \n
            labels array before merging procedure, shape (n_sample,)
        :param: merge_threshold:float \n
            threshold of dissimilarity to merge two clusters into a single one in merging procedure
        :return: labels:numpy.array \n
            labels array after merging procedure, shape (n_samples,)
    """
    m = labels.max() + 1 # current number of clusters
    i = 0
    while i < m - 1:
        flag = False
        while flag == False:
            d = np.zeros(shape=m, dtype=float) # d[j]: distance from cluster i to cluster j
            # find cluster[min_index] nearest to cluster[i]
            for j in range(i+1, m):
                d[j] = np.min(distance_matrix[np.ix_(labels == i, labels == j)])
            min_index = i + 1 + d[i+1:m].argmin()
            if d[min_index] <= merge_threshold:
                # merge nearest cluster (cluster[min_index]) into cluster[i]
                labels[labels == min_index] = i
                # rename the cluster cluster[min_index+1],...,cluster[m-1] to cluster[min_index],...,cluster[m-2]
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
    """ Reassignment procedure in refinement stage.

        :param: distance_matrix:numpy.array \n
            dissimilarity matrix of input data, array with shape (n_samples, n_sample)
        :param: labels:numpy.array \n
            labels array before reassignment procedure, shape (n_sample,)
        :return: labels:numpy.array \n
            labels array after reassignment procedure, shape (n_samples,)
    """
    n_samples = labels.shape[0] # number of samples
    n_cluster = labels.max() + 1 # number of clusters
    b = np.zeros(shape=n_samples,dtype=int) # b[i] = j => cluster j is closest to point i
    for i in range(n_samples):
        d = np.zeros(shape=n_cluster,dtype=float) # d[j] => distance from point i to cluster j
        for j in range(n_cluster):
            d[j] = distance_matrix[i, labels==j].mean()
        b[i] = d.argmin()

    return b
