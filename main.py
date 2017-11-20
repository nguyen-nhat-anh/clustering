import numpy as np
import data_processing
from sklearn import cluster

tickers = ['ACB','BID','BMP','BVH','CII','CTD','CTG','DHG','DPM','FPT',
           'GAS','GMD','HPG','HSG','KBC','KDC','MBB','MSN','MWG','NT2',
           'NVL','PVD','REE','ROS','SAB','SBT','SHB','SSI','STB','VCB',
           'VIC','VNM']
values_change = {}

for ticker in tickers:
    df = data_processing.add_value_change('include\historical-price-' + ticker + '1710201718112017.csv')
    values_change[ticker] = df['VAL_CHANGE(%)'].values

sorted_tickers = np.asarray([x[0] for x in sorted(values_change.items())])
sorted_values_change = np.asarray([x[1] for x in sorted(values_change.items())])

estimator = cluster.AffinityPropagation()
estimator.fit(sorted_values_change)

for i in range(estimator.labels_.max() + 1):
    print('Cluster %i: %s' % ((i), ', '.join(sorted_tickers[estimator.labels_ == i])))
