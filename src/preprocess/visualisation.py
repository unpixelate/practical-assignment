#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# %%

def normalise(values: pd.Series)-> pd.Series:
    '''Function that transform a series by its Min Max normalization '''
    return (values - values.min())/ (values.max() - values.min())

def plot_normalised_trends(df,columns,labels,ax):
    if not ax:
        import matplotlib.pyplot as plt
        ax = plt
    column_1,column_2 = columns
    df[column_1] = normalise(df[column_1])
    df[column_2] = normalise(df[column_2])
    df = df.sort_values('Date', ascending=True)

    if labels:
        l1,l2 = labels
        ax.plot(df['Date'], df[column_1], label=l1)
        ax.plot(df['Date'], df[column_2], label=l2)
        ax.legend(loc="lower right")
    else:
        ax.plot(df['Date'], df[column_1],df['Date'], df[column_2])
        ax.ylim(-0.2, 1.2)
    

def plot_multiple_normalised_trends(df,base_col,columns,labels):
    '''Function plots a normalised line graph for multiple numerical variables in the dataframe.
    Input:
        df: pandas DataFrame
        columns: name of columns in DataFrame'''
    n_plots = len(columns)
    n_rows = max(n_plots // 3,2) 
    n_col = 3
    fig = plt.figure(figsize=(200, 40)) # controls fig size
    fig.set_size_inches(28,16)
    fig, ax = plt.subplots(n_rows, n_col,figsize=(16,8), sharex='col', sharey='row')
    # controls subplot size here ^
    print(n_rows,n_col)
    plt.subplots_adjust(left=0.30, bottom=0.20)
    for i in range(n_rows):
        for j in range(n_col):
            print(i,j,n_plots)
            if (i)*3 + (j) >= n_plots:
                break
            plot_normalised_trends(df,(base_col,columns[i*3+j]),(base_col,labels[i*3+j]),ax[i,j])
    plt.show()





# %%
def plot_corr(df,size=10):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    for (i, j), z in np.ndenumerate(corr):
        ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=60, horizontalalignment='left');
    plt.yticks(range(len(corr.columns)), corr.columns);



# %%

def plot_confusion_matrix(cm):
    import matplotlib.pyplot as plt
    from itertools import product
    fig, ax = plt.subplots()
    cmap='Blues'
    im_ = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    xlen,ylen = cm.shape
    thresh = (cm.max() + cm.min()) / xlen
    display_labels=(0,1)
    cmap_min, cmap_max = im_.cmap(0), im_.cmap(256)
    for i, j in product(range(xlen), range(xlen)):
        color = cmap_max if cm[i, j] < thresh else cmap_min
        ax.text(j, i,format(cm[i, j], '.0f'),ha="center", va="center",color=color)
    fig.colorbar(im_, ax=ax)
    ax.set(xticks=np.arange(xlen),
        yticks=np.arange(ylen),
        xticklabels=display_labels,
        yticklabels=display_labels,
        ylabel="True label",
        xlabel="Predicted label")
    ax.set_ylim((2 - 0.5, -0.5))
    plt.show()
    return None

# %%


if __name__ == "__main__":
    # 1.
    df = pd.read_csv( os.path.join(os.getcwd(),"combined_data.csv"))
    df['Date'] = pd.to_datetime(df['Date'],  infer_datetime_format=True)
    df1 = df[['Increase']].apply(pd.value_counts)
    df1.plot.bar(rot=0)

    # 2. trend 
    closings = ("Close_10year_treasury", "Close_copper", "Close_gold","Close_hk_index" ,"Close_oil", "Close_s&p", "Value_us_sgd")
    labels = ("10year_treasury", "Copper", "Gold","HK_index" ,"Crude Oil", "S&P", "SGD v USD")
    plot_multiple_normalised_trends(df,"Close",closings,labels)

    volume = ( "Volume_copper", "Volume_gold","Volume_hk_index" ,"Volume_oil", "Volume_s&p")
    labels = ( "Copper", "Gold","HK_index" ,"Crude Oil", "S&P")
    plot_multiple_normalised_trends(df,"Close",volume,labels)

    # 3. correlation plot
    plot_corr(df)
    pass