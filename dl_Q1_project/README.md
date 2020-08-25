# Avocado Price Prediction

## 1. Getting Started

###  1.1 Prerequisities
* Python 3.6
* Virtualenv
* numpy 1.19.1
* pandas 1.1.0
* scikit-learn 0.23.1
* sklearn 0.0
* statsmodels 0.11.1
* tensorflow 2.2.0  

### 1.2 Code Folder Structure
Project consists of 4 models AR, ARIMA, RNN and LSTM.
Each model is placed in its respective directory under "models" directory.

For **AR** and **ARIMA** each consists of 6 files, which are:

1. ```data_loading.py``` - To read the preprocess data.

2. ```model_architecture``` (like ar_architecture) - Function consisting of Machine Learning model.

3. ```modelname_train.py``` (like ar_train.py) - This file responsible for training the model.

4. ```modelname_test.py``` (like ar_train.py) - This file responsible for testing and evaluating the model.

5. ```modelname_main.py``` (like ar_main.py) - This is the driver program to run the model.

6. ```modelname_Metrics.txt``` (like AR_Metrics.txt) - Ouput of model evaluation.

7. ```modelname_demo.ipynb``` - This file is python notebook with all the preprocessing, training and results.


For **LSTM** and **RNN** each consists of 6 files, which are:

1. ```data_loading.py``` - To read the preprocess data.

2. ```modelname_architecture.py``` (like lstm_architecture) - Class consisting of Deep Learning model.

3. ```modelname_train.py``` (like lstm_train.py) - This file is responsible for training the model, it reads dataset using data_loading.py, and gets model returned by modelname_architecture.py, and performs training on it, and exports the trained model in same folder.

4. ```model.h5``` (like lstm_saved_model.h5) - This is the trained model exported by train.py.

5. ```modelname_test.py``` (like lstm_test.py) - This file is responsible for evaluating the model by printing the evaluation metrics to console. This code reads test data from data_loading.py, imports pre-trained model from same folder, and evaluates.

6. ```modelname_demo.ipynb``` - This file is python notebook with all the preprocessing, training and results.

7. ```modelname_Metrics.txt``` (like LSTM_Metrics.txt) - Ouput of model evaluation.

## 2. Running the codes
1. Create virtualenv and activate it.

3. Install dependencies from ```requirements.txt``` file.
```
pip install -r requirements.txt
```
4. To run the AR model:
```
python models/AR/ar_main.py
```

5. To run the ARIMA model:
```
python models/ARIMA/arima_main.py
```

6. For LSTM:

- To train the model:
```
python  models/LSTM/lstm_train.py
```
- To test and evaluate the model:
```
python  models/LSTM/lstm_test.py
```

7. For RNN:

- To train the model:
```
python  models/RNN/lstm_train.py
```
- To test and evaluate the model:
```
python  models/RNN/lstm_test.py
```


Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
