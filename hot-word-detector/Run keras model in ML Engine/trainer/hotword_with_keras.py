from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam
import numpy as np
from datetime import datetime # for filename conventions
from tensorflow.python.lib.io import file_io
import argparse
import tensorflow as tf

Tx=1998
Ty = 497
n_freq = 101

def model(input_shape):
    """
    Function creating the model's graph in Keras.
    
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)

    Returns:
    model -- Keras model instance
    """
    
    X_input = Input(shape = input_shape)
    
    ### START CODE HERE ###
    
    # Step 1: CONV layer 
    X = Conv1D(196, kernel_size=14, strides=4)(X_input)                                 # CONV1D
    X = BatchNormalization()(X)                                 # Batch normalization
    X = Activation('relu')(X)                                 # ReLu activation
    X = Dropout(0.8)(X)                                 # dropout (use 0.8)

    # Step 2: First GRU Layer
    X = GRU(units = 128, return_sequences = True)(X) # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)                                 # dropout (use 0.8)
    X = BatchNormalization()(X)                                 # Batch normalization
    
    # Step 3: Second GRU Layer 
    X = GRU(units = 128, return_sequences = True)(X)   # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)                                 # dropout (use 0.8)
    X = BatchNormalization()(X)                                  # Batch normalization
    X = Dropout(0.8)(X)                                  # dropout (use 0.8)
    
    # Step 4: Time-distributed dense layer 
    X = TimeDistributed(Dense(1, activation = "sigmoid"))(X) # time distributed  (sigmoid)

    ### END CODE HERE ###

    model = Model(inputs = X_input, outputs = X)
    return model


def main(**args):
    train_dir="gs://hotword-detector/custom_data/XY_train"
    job_dir="gs://hotword-detector"
    logs_path = job_dir + '/logs/' + datetime.now().isoformat()
    X_stream = file_io.FileIO(train_dir+'/X.npy',mode='r')
    Y_stream = file_io.FileIO(train_dir+'/Y.npy',mode='r')
    X = np.load(X_stream)
    Y = np.load(Y_stream)
    with tf.device('/device:GPU:0'):
        model_t = model(input_shape = (Tx, n_freq)) #(1998, 101)
        opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
        model_t.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
        model_t.fit(X, Y, batch_size = 20, epochs=1)
        model_t.save('model.h5')
        with file_io.FileIO('model.h5', mode='r') as input_f:
            with file_io.FileIO(job_dir+'/trained_model/model.h5', mode='w+') as output_f:
                output_f.write(input_f.read())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input Arguments
    parser.add_argument('--train-dir',help='Cloud Storage bucket or local path to training data')
    
    parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__

    main(**arguments)
	
