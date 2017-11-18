import pandas as pd
df = pd.read_csv('include\historical-price-ACB1710201718112017.csv')

value_change = df['CLOSE'][:-1].values / df['CLOSE'][1:].values - 1
print(value_change)