import tensorflow as tf
keras = tf.keras

import data_loading as load_file
import rnn_architecture as rnn_arch

def train_model(n_hidden_units=30, n_epochs=50):
    
#    Load training data from data_loading file
    train_dataset, test_dataset, _ = load_file.get_train_and_test_dataset()
    
#    Get model from rnn_architecture file
    rnn_model_class = rnn_arch.RNN_Model(hidden_units=n_hidden_units)
    rnn_model = rnn_model_class.get_model()
    
#    Add checkpoints to save best parameters   
#    model_checkpoint = keras.callbacks.ModelCheckpoint(
#        "rnn_checkpoint", save_best_only=True)
        
#    Early stopping call back in case model does not improve for 30 epochs
    early_stopping = keras.callbacks.EarlyStopping(patience=30)
        
#    Training the model
    rnn_history = rnn_model.fit(train_dataset, epochs=n_epochs,
              validation_data=test_dataset ,
              callbacks=[early_stopping]) 
#    model_checkpoint
    
    print("\r\nCompleted Traning, Saving the model")
        
#    saving the trained model
    rnn_model.save("rnn_saved_model.h5")
    
train_model(n_hidden_units=30,n_epochs=5)