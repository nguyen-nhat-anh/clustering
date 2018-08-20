import argparse
import numpy as np
import data_processing
import clustering_algorithm
import cluster_validation
from sklearn import cluster
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='VN30 Stocks clustering program',
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-d', '--date_range', choices=['one', 'quarter', 'half'],
                    help='Date range (months): one,quarter,half\nDefault: one', required=False, default='one')
parser.add_argument('-a', '--algorithm', choices=['hierarchical', 'affinity_propagation', 'mbsas'],
                    help='Cluster algorithm: hierarchical, affinity_propagation, mbsas',required=True)
parser.add_argument('-v', '--validation', choices=['none', 'dunn', 'silhouette'],
                    help='Cluster validation options: none, dunn, silhouette\nDefault: none', required=False, default='none')
args = parser.parse_args()

tickers = ['BID','BMP','BVH','CII','CTD','CTG','DHG','DPM','FPT',
           'GAS','GMD','HPG','HSG','KBC','KDC','MBB','MSN','MWG','NT2',
           'NVL','PVD','REE','ROS','SAB','SBT','SSI','STB','VCB',
           'VIC','VNM']
values_change = {}

if args.date_range == 'one':
    for ticker in tickers:
        df = data_processing.add_value_change('include/data_1_month/historical-price-'
                                              + ticker + '1710201718112017.csv')
        values_change[ticker] = df['VAL_CHANGE(%)'].values
elif args.date_range == 'quarter':
    for ticker in tickers:
        df = data_processing.add_value_change('include/data_3_months/historical-price-'
                                              + ticker + '1708201718112017.csv')
        values_change[ticker] = df['VAL_CHANGE(%)'].values
elif args.date_range == 'half':
    for ticker in tickers:
        df = data_processing.add_value_change('include/data_6_months/historical-price-'
                                              + ticker + '1705201718112017.csv')
        values_change[ticker] = df['VAL_CHANGE(%)'].values

sorted_tickers = np.asarray([x[0] for x in sorted(values_change.items())])
sorted_values_change = np.asarray([x[1] for x in sorted(values_change.items())])

# clustering process
if args.algorithm == 'affinity_propagation':
    mean_subtracted = sorted_values_change - np.mean(sorted_values_change,axis=1,keepdims=True)
    estimator = cluster.AffinityPropagation(affinity='precomputed')
    estimator.fit(cosine_similarity(mean_subtracted))
    labels = estimator.labels_

if args.algorithm == 'hierarchical':
    dist_matrix = pdist(sorted_values_change,metric='correlation')
    Z = linkage(dist_matrix,method='ward')
    labels = fcluster(Z, t=1.25, criterion='distance')
    labels = labels - 1
    dendrogram(Z,labels=sorted_tickers)

if args.algorithm == 'mbsas':
    labels = clustering_algorithm.modified_bsas(sorted_values_change, threshold=0.9, max_n_clusters=10, refine=True,
                                                merge_threshold=0.1)

# display result
n_clusters = labels.max() + 1
for i in range(n_clusters):
    print('Cluster %i: %s' % (i, ', '.join(sorted_tickers[labels == i])))

if args.algorithm == 'hierarchical':
    plt.show()

# cluster validation
if args.validation == "dunn":
    cluster_validation.compute_dunn_index(sorted_values_change, labels, n_clusters)

if args.validation == "silhouette":
    cluster_validation.compute_silhouette(sorted_values_change, labels, n_clusters, mode='average')
    cluster_validation.plot_silhouette(sorted_values_change, labels, n_clusters)
