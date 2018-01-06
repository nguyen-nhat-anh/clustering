import numpy as np
import data_processing
from sklearn import cluster
from sklearn.metrics.pairwise import cosine_similarity

tickers = ['BID','BMP','BVH','CII','CTD','CTG','DHG','DPM','FPT',
           'GAS','GMD','HPG','HSG','KBC','KDC','MBB','MSN','MWG','NT2',
           'NVL','PVD','REE','ROS','SAB','SBT','SSI','STB','VCB',
           'VIC','VNM']
values_change = {}

for ticker in tickers:
    df = data_processing.add_value_change('include/data_1_month/historical-price-' + ticker + '1710201718112017.csv')
    # df = data_processing.add_value_change('include/data_3_months/historical-price-' + ticker + '1708201718112017.csv')
    # df = data_processing.add_value_change('include/data_6_months/historical-price-' + ticker + '1705201718112017.csv')
    values_change[ticker] = df['VAL_CHANGE(%)'].values

# export data to csv file
#import pandas as pd
#vlc = pd.DataFrame.from_dict(values_change,orient='index')
#vlc = vlc.sort_index()
#vlc.to_csv('include/value_change_pct.csv',encoding='utf-8')


sorted_tickers = np.asarray([x[0] for x in sorted(values_change.items())])
sorted_values_change = np.asarray([x[1] for x in sorted(values_change.items())])

mean_subtracted = sorted_values_change - np.mean(sorted_values_change,axis=1,keepdims=True)
estimator = cluster.AffinityPropagation(affinity='precomputed')
estimator.fit(cosine_similarity(mean_subtracted))
labels = estimator.labels_
n_clusters = labels.max() + 1
for i in range(n_clusters):
    print('Cluster %i: %s' % (i, ', '.join(sorted_tickers[labels == i])))

# from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
# from scipy.spatial.distance import pdist
# from matplotlib import pyplot as plt
# dist_matrix = pdist(sorted_values_change,metric='correlation')
# Z = linkage(dist_matrix,method='ward')
# labels = fcluster(Z, t=1.25, criterion='distance')
# labels = labels - 1
# n_clusters = labels.max() + 1
# dendrogram(Z,labels=sorted_tickers)
# plt.show()

# from clustering_algorithm.sequential import *
# labels = modified_bsas(sorted_values_change, threshold=1,max_n_clusters=10,refine=True,merge_threshold=0.1)
# n_clusters = labels.max() + 1

# from clustering_algorithm.k_means import *
# n_clusters=3
# _,labels = k_means(sorted_values_change,n_clusters)

# from clustering_algorithm.sequential import *
# labels = modified_bsas(sorted_values_change,0.9,10,refine=True,merge_threshold=0.1)
# n_clusters = labels.max() + 1
# for i in range(n_clusters):
#     print('Cluster %i: %s' % (i, ', '.join(sorted_tickers[labels == i])))

from cluster_validation.silhouette import *
plot_silhouette(sorted_values_change,labels,n_clusters,metric='correlation')
score = compute_silhouette(sorted_values_change,labels,n_clusters,metric='correlation')
for i in zip(sorted_tickers,score):
    print(i)
