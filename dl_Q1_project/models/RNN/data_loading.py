import numpy as np
import pandas as pd
import tensorflow as tf


# For time series, creating a window of provided window size with shift of provided value
def create_window_dataset(series, window_size,shift_val=1):
    
    ds = tf.data.Dataset.from_tensor_slices(series,)
    ds = ds.window(window_size + 1, shift=shift_val, drop_remainder=True)
    ds = ds.flat_map(lambda window: window.batch(window_size + 1))
    ds = ds.map(lambda window: (window[:-1], window[1:]))
    return ds.batch(1).prefetch(1)


# Returns train dataset and test dataset and also a dataframe with test data with region column,
# so that while testing the model, test data for each region can provided separately
def get_train_and_test_dataset():
    
#    Read data and convert date column from Object to date
    df = pd.read_csv('https://raw.githubusercontent.com/mdrkb/avocado-prices/master/avocado_updated.csv', 
                 parse_dates=['Date'])
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.drop("Unnamed: 0",axis=1)
        
    regions = ['West', 'Northeast', 'Southeast', 'SouthCentral','Midsouth']

#    Filter the dataset to get only 5 regions and sort it based on Date and set date as index
    region_df = df[df["region"].isin(regions)]
    region_df = region_df[region_df["type"]=="organic"][["Date","AveragePrice","region"]]
    region_df = region_df.sort_values(["region","Date"])
    region_df = region_df.set_index("Date")
    
#    Convert region into one-hot encoded values
    region_df_with_region = pd.concat([region_df,pd.get_dummies(region_df.region)], axis=1)
    
    train_dataset_list = []
    test_dataset_list = []
    forecast_df = region_df_with_region[0:0]
    
#    Split the dataset into train data and test data, for each region.
#    For each region 80% is put in train_data, 20% is put in test data and same test data is
#    also saved in dataframe "forecast_df" to be used during testing, since it has region 
#    information 
    for reg in regions:
        reg_df = region_df_with_region[region_df_with_region.region == reg]
        train_size = int(np.floor(reg_df.AveragePrice.count() * 0.8)) 
        reg_train_df = reg_df[:train_size]
        reg_test_df = reg_df[train_size:]
        reg_train_series = reg_train_df.drop(columns=["region"]).values
        reg_test_series = reg_test_df.drop(columns=["region"]).values

        reg_train_ds = create_window_dataset(reg_train_series,10,shift_val=2)
        train_dataset_list.append(reg_train_ds)
        reg_test_ds = create_window_dataset(reg_test_series,10,shift_val=5)
        test_dataset_list.append(reg_test_ds)
        
        forecast_df = forecast_df.append(reg_test_df)
    
    test_dataset = None
    train_dataset = None
   
#   Concatenate dataset for each region to form single train_dataset with training data for all
#    regions, and single test_dataset with test data for all regions
    count=0
    for i in train_dataset_list:
        if count == 0:
            train_dataset = i
        else:
            train_dataset = train_dataset.concatenate(i)
        count = count + 1
    
    count=0 
    for i in test_dataset_list:
        if count == 0:
            test_dataset = i
        else:
            test_dataset = test_dataset.concatenate(i)
        count = count + 1
        
    count = 0
    for i,j in train_dataset.as_numpy_iterator():
        count  = count + 1
        
    return [train_dataset,test_dataset,forecast_df]