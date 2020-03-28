#%%
import os
import pandas as pd
from pathlib import Path
import statsmodels.api as sm
import numpy as np
from preprocess.aggregator import LaggedDataFrame, get_data
from preprocess.visualisation import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# linear regression
df = get_data()
df_lagged = LaggedDataFrame(df,exclude_from_lagging=["Date", "Increase", "Open"]).buildLaggedFeatures(lag=2)
del df_lagged["Close"]
del df_lagged["Open"]


y = df_lagged['Increase']
#df[df.columns[~df.columns.isin(['C','D'])]]
X = df_lagged[df_lagged.columns[~df_lagged.columns.isin(['Increase','Date'])]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2020)

# %%
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
plot_confusion_matrix(cm)
# %%


# %%
