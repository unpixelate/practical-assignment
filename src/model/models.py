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
from preprocess.aggregator import LaggedDataFrame, get_data
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LSTM, Dropout,RepeatVector
from sklearn.model_selection import train_test_split
import attention 
import importlib
importlib.reload(attention)
from attention import attention_3d_block
from sklearn.preprocessing import MinMaxScaler
df = get_data().copy()
df = df[df.columns[~df.columns.isin(['Date',"Close","Open"])]]


time_step = 10
data_size, data_dim = df.shape
debug(data_size, data_dim)
data_resize = data_size//time_step
debug(data_resize)
data_size_subset = data_resize * time_step 
debug(data_size_subset)
batch_size = 10
epoch = 100
drop =0.2   

y = df['Increase'][:data_size_subset]
X = df[df.columns[~df.columns.isin(['Increase'])]][:data_size_subset]

X_length , input_dim = X.shape
debug(X.shape)
#scalar = MinMaxScaler((0,1))
#X = scalar.fit_transform(X)
# required format for lstm
X_shaped = X.values.reshape(data_resize,time_step,input_dim) # 1= timesteps
debug(X.shape)
Y_shaped = y.values.reshape(data_resize,time_step)
Y_shaped = Y_shaped[:,-1]
debug("Y_shape: ",Y_shaped.shape)
debug(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X_shaped, Y_shaped, test_size=0.2)

from keras.layers import concatenate
from keras import backend as K
from tensorflow.keras.layers import Dense, Lambda, dot, Activation, concatenate,GlobalAveragePooling1D
def get_model(time_step,input_dim):
    i = Input(shape=(time_step, input_dim))
    x = LSTM(216, return_sequences=True)(i)
    x = Dropout(0.2)(x)
    x = LSTM(128, return_sequences=True)(x)
    x = LSTM(100, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(128)(x)
    x = Dense(time_step, activation='relu')(x)
    model = Model(inputs=[i], outputs=[x])
    model.compile(loss='binary_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])
    print(model.summary())
    return model

from keras.callbacks import EarlyStopping,ModelCheckpoint   ,ReduceLROnPlateau        
earlyStopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='min')
mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.01, patience=4, verbose=1, epsilon=1e-4, mode='min')

baseline = get_model(time_step,input_dim)
# Train the model
baseline.fit(X_train, y_train,
          batch_size= batch_size, epochs=100,
          validation_data = (X_test, y_test),
          callbacks=[earlyStopping]  )


#Evalute the model
loss, acc = baseline.evaluate(X_test, y_test,1)

print("Keras: \n%s: %.2f%%" % (baseline.metrics_names[1], acc*100))


#%%
y_pred =  (baseline.predict(X_test)>0.5).reshape(y_test.shape)
cm = confusion_matrix(y_test,y_pred)
plot_confusion_matrix(cm)
print("Accuracy: {:.3f}".format(get_accuracy(y_test, y_pred)[0]))# %%
print(y_pred)



# %%
