from statsmodels.tsa.ar_model import AR


def autoregressive(train_data):
    model = AR(train_data)
    return model