from data_loading import *
from ar_train import *
from ar_test import *


def main_AR(data="https://raw.githubusercontent.com/mdrkb/avocado-prices/master/avocado_updated.csv"):
    with open('AR_Metrics.txt', 'w+') as fileObj:
        df = read_data(data)
        df = preprocess_data(df)
        df_US, df_regions, df_subregions = split_df_by_regions(df)

        print("-"*15, "TotalUS", "-"*15)
        # Get time series for conventional avocado for totalUS
        df_US_conventional = df_US[df_US['type_organic']==0]["AveragePrice"]
        df_US_conventional = make_stationary(df_US_conventional)
        train, test, split_range = train_test_split(df_US_conventional)
        model_fitted = train_AR(train)
        pred = test_AR(model_fitted, split_range, len(df_US_conventional)-1, test)
        fileObj.write("TotalUS Conventional:\n")
        fileObj.write(str(get_evaluation_metric(test, pred))+'\n\r')

        # Get time series for organic avocado for totalUS 
        df_US_organic = df_US[df_US['type_organic']==1]["AveragePrice"]
        df_US_organic = make_stationary(df_US_organic)
        train, test, split_range = train_test_split(df_US_organic)
        model_fitted = train_AR(train)
        pred = test_AR(model_fitted, split_range, len(df_US_organic)-1, test)
        fileObj.write("TotalUS Organic:\n")
        fileObj.write(str(get_evaluation_metric(test, pred))+'\n\r')

        
        regions = ['West', 'Midsouth', 'Northeast', 'SouthCentral', 'Southeast']
        for region in regions:
            print("-"*15, region, "-"*15)
            # Get time series for conventional avocado for a region
            df_region_conventional = df_regions[(df_regions['type_organic']==0) & (df_regions['region']==region)]["AveragePrice"]
            df_region_conventional = make_stationary(df_region_conventional)
            train, test, split_range = train_test_split(df_region_conventional)
            model_fitted = train_AR(train)
            pred = test_AR(model_fitted, split_range, len(df_region_conventional)-1, test)
            fileObj.write(f"{region} Conventional:\n")
            fileObj.write(str(get_evaluation_metric(test, pred))+'\n\r')
            
            # Get time series for organic avocado for a region
            df_region_organic = df_regions[(df_regions['type_organic']==1) & (df_regions['region']==region)]["AveragePrice"]
            df_region_organic = make_stationary(df_region_organic)
            train, test, split_range = train_test_split(df_region_organic)
            model_fitted = train_AR(train)
            pred = test_AR(model_fitted, split_range, len(df_region_organic)-1, test)
            fileObj.write(f"{region} Organic:\n")
            fileObj.write(str(get_evaluation_metric(test, pred))+'\n\r')


if __name__ == "__main__":
    main_AR()