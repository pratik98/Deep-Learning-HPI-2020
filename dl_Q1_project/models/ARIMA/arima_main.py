from data_loading import *
from arima_train import *
from arima_test import *


def main_ARIMA(data="https://raw.githubusercontent.com/mdrkb/avocado-prices/master/avocado_updated.csv"):
    with open('ARIMA_Metrics.txt', 'w+') as fileObj:
        df = read_data(data) 
        df = preprocess_data(df)
        df_US, df_regions, df_subregions = split_df_by_regions(df)

        print("-"*15, "TotalUS", "-"*15)
        # Get time series for conventional avocado for totalUS
        df_US_conventional = df_US[df_US['type_organic']==0]["AveragePrice"]
        train, test, split_range = train_test_split(df_US_conventional)
        # grid_serach_arima(train) # Grid search for parameters
        model_fitted = train_ARIMA(train, *(3,0,4))
        pred = test_ARIMA(model_fitted, len(test), test)
        fileObj.write("TotalUS Conventional:\n")
        fileObj.write(str(get_evaluation_metric(test, pred))+'\n\r')

        # Get time series for organic avocado for totalUS 
        df_US_organic = df_US[df_US['type_organic']==1]["AveragePrice"]
        train, test, split_range = train_test_split(df_US_organic)
        # grid_serach_arima(train) # Grid search for parameters
        model_fitted = train_ARIMA(train, *(4,0,3))
        pred = test_ARIMA(model_fitted, len(test), test)
        fileObj.write("TotalUS Organic:\n")
        fileObj.write(str(get_evaluation_metric(test, pred))+'\n\r')

        regions = ['West', 'Midsouth', 'Northeast', 'SouthCentral', 'Southeast']
        params = {
            'West': [(3,0,2), (2,0,3)], 
            'Midsouth': [(3,0,4), (4,0,2)], 
            'Northeast': [(4,0,2), (2,0,3)], 
            'SouthCentral': [(1,0,0), (3,0,0)], 
            'Southeast': [(2,0,0),(1,0,1)]
        }

        for region in regions:
            print("-"*15, region, "-"*15)
            # Get time series for conventional avocado for a region
            df_region_conventional = df_regions[(df_regions['type_organic']==0) & (df_regions['region']==region)]["AveragePrice"]
            train, test, split_range = train_test_split(df_region_conventional)
            # grid_serach_arima(train) # Grid search for parameters
            model_fitted = train_ARIMA(train, *params[region][0])
            pred = test_ARIMA(model_fitted, len(test), test)
            fileObj.write(f"{region} Conventional:\n")
            fileObj.write(str(get_evaluation_metric(test, pred))+'\n\r')
            
            # Get time series for organic avocado for a region
            df_region_organic = df_regions[(df_regions['type_organic']==1) & (df_regions['region']==region)]["AveragePrice"]
            train, test, split_range = train_test_split(df_region_organic)
            # grid_serach_arima(train) # Grid search for parameters
            model_fitted = train_ARIMA(train, *params[region][1])
            pred = test_ARIMA(model_fitted, len(test), test)
            fileObj.write(f"{region} Organic:\n")
            fileObj.write(str(get_evaluation_metric(test, pred))+'\n\r')


if __name__ == "__main__":
    main_ARIMA()