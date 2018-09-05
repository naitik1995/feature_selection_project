# %load q02_best_k_features/build.py
# Default imports

import pandas as pd
import numpy as np
data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression


# Write your solution here:
def percentile_k_features(df,k=20):
    features=df.iloc[:,:-1]
    target=df['SalePrice']
    res = SelectPercentile(f_regression,percentile=20)
    X_new=res.fit_transform(features, target)
    best_k={}
    scores= res.scores_
    tmp=np.sort(scores)
    for i in range(len(scores)):
        best_k[scores[i]]=i
    best_f=[]
    for i in range(7):
        best_f.append(df.columns[best_k[tmp[::-1][i]]])
    return best_f



