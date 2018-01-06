import argparse
import numpy as np
import data_processing
from sklearn import cluster
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='VN30 Stocks clustering program')
parser.add_argument('-d', '--date_range', help='Date range (months): one,quarter,half', required=True, default='one')
parser.add_argument('-a', '--algorithm', help='Cluster algorithm: hierarchical, affinity_propagation',required=True)
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

if args.algorithm == 'affinity_propagation':
    mean_subtracted = sorted_values_change - np.mean(sorted_values_change,axis=1,keepdims=True)
    estimator = cluster.AffinityPropagation(affinity='precomputed')
    estimator.fit(cosine_similarity(mean_subtracted))
    for i in range(estimator.labels_.max() + 1):
        print('Cluster %i: %s' % (i, ', '.join(sorted_tickers[estimator.labels_ == i])))

if args.algorithm == 'hierarchical':
    dist_matrix = pdist(sorted_values_change,metric='correlation')
    Z = linkage(dist_matrix,method='ward')
    dendrogram(Z,labels=sorted_tickers)
    plt.show()
