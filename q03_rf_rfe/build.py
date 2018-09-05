# %load q03_rf_rfe/build.py
# Default imports
import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier


# Your solution code here
def rf_rfe(data):
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    len_features = len(X.columns) / 2
    random_est = RandomForestClassifier()
    random_est.fit(X,y)
    rfe_clf = RFE(random_est,n_features_to_select=len_features)
    X_new= rfe_clf.fit_transform(X,y)
    res = X.columns.values[rfe_clf.get_support()]
    list_op = []
    for i in res:
        list_op.append(i)
    return list_op    
rf_rfe(data)



