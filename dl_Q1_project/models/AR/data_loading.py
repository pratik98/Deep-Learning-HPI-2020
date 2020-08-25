import pandas as pd
import numpy as np
import itertools
import warnings
from sklearn import metrics
from statsmodels.tsa.stattools import adfuller
warnings.filterwarnings('ignore')



def read_data(filename):
    """
    Read the csv file and create dataframe
    """
    df = pd.read_csv(filename, parse_dates=['Date'], index_col='Date')
    df = df.sort_index()
    print(f"Shape of the dataframe: {df.shape}\n")
    return df

def preprocess_data(df):
    """
    Apply preprocessing on dataframe
    """
    # Drop unnecessary columns
    df = df.drop(["Unnamed: 0", "year"], axis=1) 
    # Apply one hot encoding
    df = pd.get_dummies(df, columns=["type"], drop_first=True) 
    return df

def make_stationary(df):
    """
    Make time series stationary
    """
    dftest = adfuller(df) # Dickey-Fuller test is used
    pvalue = dftest[1]
    critical_val_5percent = dftest[4]["5%"]
    stationary = pvalue < critical_val_5percent

    if not stationary:
        df = df.diff(periods=1)[1:]
    
    return df

def split_df_by_regions(df):
    """
    Split the dataframe based on TotalUS, regions and sub-regions
    """
    df_US = df[df.region == 'TotalUS']
    regions = ['West', 'Midsouth', 'Northeast', 'SouthCentral', 'Southeast']
    df_regions = df[df.region.apply(lambda x: x in regions and x != 'TotalUS')]
    df_subregions = df[df.region.apply(lambda x: x not in regions and x != 'TotalUS')]
    return df_US, df_regions, df_subregions

def train_test_split(df, split_percentage=0.95):
    split_range = int(np.round(len(df)*split_percentage))
    train = df[:split_range].values
    test = df[split_range:].values
    return train, test, split_range

def get_evaluation_metric(test, pred):
    return {
        "MAE": metrics.mean_absolute_error(test, pred),
        "MSE": metrics.mean_squared_error(test, pred),
        "RMSE": np.sqrt(metrics.mean_absolute_error(test, pred))
    }