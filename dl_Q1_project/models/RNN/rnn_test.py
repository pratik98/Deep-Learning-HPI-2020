from tensorflow import keras
import numpy as np
import data_loading as load_file

# Method to get metrics from the model
def get_metrics_for_RNN(rnn_model,test_set):
    metric_val = rnn_model.evaluate(test_set)
    metric_string = "\r\n  Loss is - " + str(metric_val[0]) + ",\r\n  Mean Absolute Error is  -  " +  str(metric_val[2]) + \
    ",\r\n  Mean Squared Error is  -  " + str(metric_val[1]) \
    + ",\r\n  Root Mean Squared Error is  -  "       +   str(np.sqrt(metric_val[1]))
    return metric_string

# Evaluate the model
def test_model():

#    Read the test data from data_loading file, and split it into 5 regions.
    _, test_dataset, forecast_df = load_file.get_train_and_test_dataset()    
    forecast_regions = list(forecast_df.region.unique())
    
#    Read the pre-trained model    
    rnn_model = keras.models.load_model("rnn_saved_model.h5")    
   
    # Printing Metrics for all the regions
    print("<========================= Metrics =======================>")
    fileObj = open("RNN_Metrics.txt","w+")

    for each_region in forecast_regions:
      f_df = forecast_df[forecast_df.region==each_region]
      f_series = f_df.drop("region",axis=1).values
      f_ds = load_file.create_window_dataset(f_series,13,shift_val=13)
      
      eval_metrics = get_metrics_for_RNN(rnn_model,f_ds)
      print("\r\nFor Region - ",each_region,eval_metrics)
      print("\r\n")
      
      fileObj.write("\r\nFor Region - " + each_region + eval_metrics)
      fileObj.write("\r\n")
    
    fileObj.close()



test_model()