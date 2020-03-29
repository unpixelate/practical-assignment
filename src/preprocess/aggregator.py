#%%
import pandas as pd
import os
from pathlib import Path
from typing import List

DEBUG = True
ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = os.path.join(ROOT,"data")

def debug(*msg):
    if DEBUG:
        for i in msg:
            print(i, end="")
    print()       

def rename_cols(orig_names
                ,prefix
                ,callback = lambda x: x.replace('.csv','')):
    """" @params: callback - lambda function """

    new_names = []
    for i in orig_names:
        if i == "Date":
            new_names.append('Date')
            continue
        else:
            i = i + "_" + prefix
            new_names.append(i)
    if callback:
        return list(map(callback,new_names))
    return new_names


def aggregate(folder_path=DATA_ROOT
            , files_to_be_excluded: List[str]=[]
            , columns: List[str]=[])-> pd.DataFrame:

    if not files_to_be_excluded:
        datafiles = os.listdir(folder_path)
    else:
        datafiles = list(filter(lambda x: x not in files_to_be_excluded, os.listdir(folder_path)))
    
    df = pd.read_csv(os.path.join(DATA_ROOT,datafiles[0]))[columns]
    debug(df.columns)
    df.columns = rename_cols(df.columns,datafiles[0])
    for data in datafiles[1:]:
        try:
            df_temp = pd.read_csv(os.path.join(DATA_ROOT,data))[columns]
            df_temp.columns = rename_cols(df_temp.columns,data)
            df = df.merge(df_temp,how='inner',left_on = "Date", right_on="Date")
        except Exception as e:
            debug("**Error in file: {0} and error message is {1}".format(data,e))
            continue    
    orig_length = len(df)
    df = df.dropna()
    debug("No. of rows with NA: ", orig_length-len(df))
    debug("Total no. of rows: ", len(df))
    debug(df.head())
    return df

#df = aggregate(files_to_be_excluded=["us_sgd.csv",'sti.csv'],columns=["Date", "Close", "Volume"])

#df_temp = pd.read_csv(os.path.join(DATA_ROOT,"us_sgd.csv"))[["Date", "Value"]]
#df_temp.columns = rename_cols(df_temp.columns,"us_sgd")
#df = df.merge(df_temp,how='inner',left_on = "Date", right_on="Date")

#df_sti = pd.read_csv(os.path.join(DATA_ROOT,"sti.csv"))[["Date", "Open","Close"]]
#df_sti["Increase"] = df_sti["Close"] > df_sti["Open"]
#df_sti["Open"] 
#df = df.merge(df_sti,how='inner',left_on = "Date", right_on="Date")
#df['Date'] = pd.to_datetime(df['Date'],  infer_datetime_format=True)
#df.to_csv('combined_data.csv', index=False)

# %%
class LaggedDataFrame:
    def __init__(self, df, exclude_from_lagging):
        self.exclude_from_lagging = exclude_from_lagging
        self.df = df
        self.transformed_df = None

    def __repr__(self):
        if not self.transformed_df:
            return "Not transformed"
        return self.transformed_df.head()
    def buildLaggedFeatures(self, lag=2, dropna=True):
        '''
        https://stackoverflow.com/questions/20410312/how-to-create-a-lagged-data-structure-using-pandas-dataframe
        Builds a new DataFrame to facilitate regressing over all possible lagged features
        '''
        if self.transformed_df:
            return self.transformed_df

        if isinstance(self.df,pd.DataFrame):
            new_dict={}
            for col_name in self.df:
                new_dict[col_name]=self.df[col_name]
                if col_name in self.exclude_from_lagging:
                    continue
                # create lagged Series
                for l in range(1,lag+1):

                    new_dict['%s_lag%d' %(col_name,l)]=self.df[col_name].shift(l)
            
            res=pd.DataFrame(new_dict,index=self.df.index)

        elif isinstance(self.df,pd.Series):
            the_range=range(lag+1)
            res=pd.concat([self.df.shift(i) for i in the_range],axis=1)
            res.columns=['lag_%d' %i for i in the_range]
        else:
            raise NotImplementedError
        if dropna:
            res.dropna()
        self.transformed_df = res.iloc[lag:]
        return self.transformed_df

#df_lagged = LaggedDataFrame(df,exclude_from_lagging=["Date", "Increase"]).buildLaggedFeatures(lag=2)
#df_lagged.to_csv('lagged_combined_data.csv',index=False)

# %%
def get_data():
    ROOT = os.getcwd()
    df = pd.read_csv( os.path.join(os.path.join(ROOT,"combined_data.csv")))
    df['Date'] = pd.to_datetime(df['Date'],  infer_datetime_format=True)
    return df