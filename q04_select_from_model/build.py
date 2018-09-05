# %load q04_select_from_model/build.py
# Default imports
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')


# Your solution code here
def select_from_model(data):
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    rf_clf = RandomForestClassifier()
    rf_clf.fit(X,y)
    model_select = SelectFromModel(rf_clf, prefit=True)
    X_new = model_select.transform(X)
    cols_data = X.columns.values[model_select.get_support()]
    cols_list = []
    for i in cols_data:
        cols_list.append(i)
    return cols_list    


