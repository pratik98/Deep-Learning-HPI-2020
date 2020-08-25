import warnings
warnings.filterwarnings('ignore')
from arima_architecture import arima


def train_ARIMA(train_data, *param):
    model = arima(train_data, *param)
    model_fitted = model.fit()
    return model_fitted