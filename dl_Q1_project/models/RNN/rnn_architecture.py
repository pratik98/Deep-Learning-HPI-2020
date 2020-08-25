import tensorflow as tf
keras = tf.keras

# class returning a RNN model with 2 RNN layer and 1 Dense Layer
class RNN_Model():
    
    def __init__(self,hidden_units):
        super(RNN_Model, self).__init__()
        self.rnn_layer_1 = keras.layers.SimpleRNN(hidden_units, return_sequences=True,input_shape=[None,6])
        self.rnn_layer_2 = keras.layers.SimpleRNN(hidden_units, return_sequences=True)
        self.dense_layer =keras.layers.Dense(6)
    
    def get_model(self):
#        rnn_model = keras.models.Sequential([
#        self.rnn_layer_1, self.rnn_layer_2,self.dense_layer])

        rnn_model = keras.models.Sequential([keras.layers.SimpleRNN(30, return_sequences=True,input_shape=[None,6]),
        keras.layers.SimpleRNN(30, return_sequences=True),
        keras.layers.Dense(6)])
    
        optimizer = keras.optimizers.Adam(lr=1e-3)

        rnn_model.compile(loss='mse',
              optimizer=optimizer,
              metrics=["mse","mae"])

        return rnn_model