def test_AR(model_fitted, start, end, test_data):
    pred = model_fitted.predict(start=start, end=end)
    print('The lag value is: %s' % model_fitted.k_ar)
    print('The coefficients of the model are:\n %s\n' % model_fitted.params)
    return pred