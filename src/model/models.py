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

def get_lagged_train_test_data(lag):
    df = get_data().copy()
    df_lagged = LaggedDataFrame(df,exclude_from_lagging=["Date", "Increase", "Open"]).buildLaggedFeatures(lag=lag)
    del df_lagged["Close"]
    del df_lagged["Open"]
    y = df_lagged['Increase']
    #df[df.columns[~df.columns.isin(['C','D'])]]
    X = df_lagged[df_lagged.columns[~df_lagged.columns.isin(['Increase','Date'])]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2020)
    return  X_train, X_test, y_train, y_test 

def get_accuracy(y_true,y_pred):
    return sum(y_true==y_pred)/len(y_true),

# %%
from sklearn import tree

X_train, X_test, y_train, y_test = get_lagged_train_test_data(5)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
plot_confusion_matrix(cm)
print("Accuracy: {0}".format(get_accuracy(y_test, y_pred)))

X_train, X_test, y_train, y_test = get_lagged_train_test_data(10)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
plot_confusion_matrix(cm)
print("Accuracy: {0}".format(get_accuracy(y_test, y_pred)))


X_train, X_test, y_train, y_test = get_lagged_train_test_data(15)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
plot_confusion_matrix(cm)
print("Accuracy: {0}".format(get_accuracy(y_test, y_pred)))


# %%
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
clf = RandomForestClassifier( n_estimators=1000, max_depth=5, random_state=2020)
X_train, X_test, y_train, y_test = get_lagged_train_test_data(10)
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
plot_confusion_matrix(cm)
print("Accuracy: {0}".format(get_accuracy(y_test, y_pred)))

# %%
import matplotlib.pyplot as plt
tree_feature_importances = clf.feature_importances_
feature_names = X_train.columns
sorted_idx = tree_feature_importances.argsort()
y_ticks = np.arange(0, 20)
fig, ax = plt.subplots()
ax.barh(y_ticks, tree_feature_importances[sorted_idx][len(feature_names)-20:])
ax.set_yticklabels(feature_names[sorted_idx][len(feature_names)-20:])
ax.set_yticks(y_ticks)
ax.set_title("Random Forest Feature Importances (MDI)")
fig.tight_layout()
plt.show()

# %%
from xgboost import XGBClassifier   
clf = XGBClassifier( n_estimators=1000, max_depth=7, random_state=2020)
X_train, X_test, y_train, y_test = get_lagged_train_test_data(15)
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
plot_confusion_matrix(cm)
print("Accuracy: {0}".format(get_accuracy(y_test, y_pred)))


# %%
DEBUG=True
def debug(*msg):
    if DEBUG:
        print(*msg)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split

df = get_data().copy()
df = df[df.columns[~df.columns.isin(['Date',"Close","Open"])]]


time_step = 15
data_size, data_dim = df.shape
debug(data_size, data_dim)
data_resize = data_size//time_step
debug(data_resize)
data_size_subset = data_resize * time_step 
debug(data_size_subset)
batch_size = 1
epoch = 100
drop =0.2   

y = df['Increase'][:data_size_subset]
X = df[df.columns[~df.columns.isin(['Increase'])]][:data_size_subset]
X_length , X_dim = X.shape
debug(X.shape)

# required format for lstm
X_shaped = X.values.reshape(data_resize,time_step,X_dim) # 1= timesteps
debug(X.shape)
Y_shaped = y.values.reshape(data_resize,time_step)
debug(X.shape)

model = Sequential()
model.add(LSTM(36, return_sequences=True, input_shape=(1, data_dim)))  # returns a sequence of vectors of dimension 40
model.add(Dropout(drop))
model.add(LSTM(24,return_sequences=True))  # returns a sequence of vectors of dimension 40
model.add(Dropout(drop))
model.add(LSTM(10,return_sequences=True))  # returns a sequence of vectors of dimension 40
model.add(Dropout(drop))
model.add(LSTM(20))  # return a single vector of dimension 40
model.add(Dropout(drop))
model.add(Dense(time_step, activation='sigmoid')) 


# %%
