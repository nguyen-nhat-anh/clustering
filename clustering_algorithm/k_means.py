import numpy as np


def k_means(data, n_clusters, algorithm='kmeans++'):
    """ Clustering using k-means algorithm

    :param: data:numpy.array \n
        input data, array with shape (n_samples, n_features)
    :param: n_cluster:int \n
        number of clusters
    :return: centroids:numpy.array \n
        array of centroids, shape (n_cluster, n_features)
    :return: labels:numpy.array \n
        array of labels corresponding to input data, shape (n_samples,)

    Reference:  MacKay, David (2003). "Chapter 20. An Example Inference Task: Clustering"
                Information Theory, Inference and Learning Algorithms. Cambridge University Press. pp. 284–292.
    """
    centroids = _initialize(data, n_clusters, algorithm)
    while True:
        previous_centroids = centroids
        labels = _assign_labels(data, centroids)
        centroids = _update_centroids(data, labels, previous_centroids)
        if np.array_equal(centroids, previous_centroids):
            break

    return [centroids,labels]


def _initialize(data, n_clusters, algorithm):
    """ Initialize the centroids

        :param: data:numpy.array \n
            input data, array with shape (n_samples, n_features)
        :param: n_cluster:int \n
            number of clusters
        :return: centroids:numpy.array \n
            array of centroids with shape (n_clusters, n_features)
    """
    if algorithm == 'kmeans++':
        n_samples, n_features = data.shape
        centroids = np.empty(shape=(n_clusters, n_features), dtype=float)
        centroids[0] = data[np.random.choice(range(n_samples))]
        probabilities = np.empty(shape=(n_samples,), dtype=float)
        for i in range(1,n_clusters):
            for k in range(n_samples):
                probabilities[k] = _get_nearest_distance(data[k], centroids[0:i]) ** 2
            probabilities = probabilities / np.sum(probabilities)
            centroids[i] = data[np.random.choice(range(n_samples), p=probabilities)]

        return centroids
    else:
        return np.random.uniform(size=(n_clusters,data.shape[1]))


def _assign_labels(data, centroids):
    """ Assign data points to their nearest centroid

    :param: data:numpy.array \n
        input data, array with shape (n_samples, n_features)
    :param: centroids:numpy.array \n
        array of centroids with shape (n_clusters, n_features)
    :return: labels:numpy.array \n
        array with shape (n_samples,) \n
        labels[i] = k means data[i] belongs to cluster k-th
    """
    n_samples,n_features = data.shape
    labels = np.empty(shape=(n_samples), dtype=int)
    for i in range(n_samples):
        k = _get_nearest_centroid(data[i], centroids)
        labels[i] = k

    return labels


def _get_nearest_centroid(data_point, centroids):
    """ Get index of the centroid nearest to a given data point

    :param: data_point:numpy.array \n
        input data point, shape (n_features,)
    :param: centroids:numpy.array \n
        array of centroids with shape (n_clusters, n_features)
    :return: index:int \n
        return index where centroids[index] is the nearest point
    """
    index = 0
    min_value = np.linalg.norm(data_point - centroids[0])
    n_clusters = centroids.shape[0]
    for i in range(n_clusters):
        if min_value > np.linalg.norm(data_point - centroids[i]):
            min_value = np.linalg.norm(data_point - centroids[i])
            index = i

    return index


def _get_nearest_distance(data_point, centroids):
    """ Return the shortest distance from a data point to closest centroid we have already chosen

    :param data_point:numpy.array \n
        shape (n_features,)
    :param centroids:numpy.array \n
        a set of centroids, array with shape (n_centroids,n_features)
    :return: float
        the shortest distance from data point to closest centroid
    """
    n_centroids = centroids.shape[0]
    distances = np.linalg.norm(data_point - centroids,axis=1)
    return np.min(distances)


def _update_centroids(data, labels, old_centroids):
    """ Adjust the centroids to match the sample means of the data points that they are responsible for

    :param: data:numpy.array \n
        input data, array with shape (n_samples, n_features)
    :param: labels:numpy.array \n
        array of labels corresponding to input data, shape (n_samples,)
    :param: old_centroids:labels:numpy.array \n
        array of old centroids with shape (n_clusters, n_features)
    :return:updated_centroids:numpy.array \n
        array of updated centroids with shape (n_clusters, n_features)
    """
    n_clusters, n_features = old_centroids.shape
    updated_centroids = np.empty(old_centroids.shape, dtype=float)
    for i in range(n_clusters):
        if data[labels == i].size == 0:
            updated_centroids[i] = old_centroids[i]
        else:
            updated_centroids[i] = np.mean(data[labels == i], axis=0)

    return updated_centroids

