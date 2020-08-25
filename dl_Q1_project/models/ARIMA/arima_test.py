def test_ARIMA(model_fitted, test_length, test_data):
    pred = model_fitted.forecast(steps=test_length)[0]
    print('The lag value is: %s' % model_fitted.k_ar)
    print('The coefficients of the model are:\n %s\n' % model_fitted.params)
    return pred