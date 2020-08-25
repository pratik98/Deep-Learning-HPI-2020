import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.arima_model import ARIMA as arima_model


def arima(train_data, *param):
    model = arima_model(train_data, order=param)
    return model