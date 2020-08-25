import warnings
warnings.filterwarnings('ignore')
from ar_architecture import autoregressive


def train_AR(train_data):
    model = autoregressive(train_data)
    model_fitted = model.fit()
    return model_fitted