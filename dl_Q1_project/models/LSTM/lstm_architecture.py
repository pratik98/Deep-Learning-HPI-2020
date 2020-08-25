import tensorflow as tf
keras = tf.keras

# class returning a LSTM model with 1 LSTM layer and 1 Dense Layer
class LSTM_Model():
    
    def __init__(self,hidden_units):
        super(LSTM_Model, self).__init__()
        self.layer_1 = keras.layers.LSTM(hidden_units,batch_input_shape=[1,None,6],name="first",return_sequences=True)
        self.layer_2 = keras.layers.Dense(6,name="second")
    
    def get_model(self):
        lstm_model = keras.models.Sequential([
        self.layer_1 ,
        self.layer_2
        ])
        
        optimizer = keras.optimizers.Adam(lr=1e-2)

        lstm_model.compile(loss='mse',
              optimizer=optimizer,
              metrics=["mse","mae"])
        return lstm_model