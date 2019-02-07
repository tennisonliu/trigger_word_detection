import numpy as np
from pydub import AudioSegment
import random
import sys
import io
import os
import glob
import IPython
from td_utils import *
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam

def f1(y_true, y_pred):
	def recall(y_true, y_pred):
		true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
		possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
		recall = true_positives / (possible_positives + K.epsilon())
		return recall

	def precision(y_true, y_pred):
		true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
		predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
		precision = true_positives / (predicted_positives + K.epsilon())
		return precision
	precision = precision(y_true, y_pred)
	recall = recall(y_true, y_pred)
	return 2*((precision*recall)/(precision+recall+K.epsilon()))

def model(input_shape):
    X_input = Input(shape = input_shape)
    
    # Conv Layer
    X = Conv1D(196, kernel_size=15, strides = 4)(X_input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Dropout(0.8)(X)
    
    # GRU Layer
    X = GRU(units = 128, return_sequences = True)(X)
    X = Dropout(0.8)(X)
    X = BatchNormalization()(X)
    
    # GRU Layer
    X = GRU(units = 128, return_sequences = True)(X)
    X = Dropout(0.8)(X)
    X = BatchNormalization()(X)
    X = Dropout(0.8)(X)
    
    # Time-distributed Dense Layer
    X = TimeDistributed(Dense(1, activation = "sigmoid"))(X)
    
    model = Model(inputs = X_input, outputs = X)
    return model

def train_model(X_train_path, Y_train_path):
	opt = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, decay = 0.01)
	model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ["accuracy"]
	X = np.load(X_train_path)
	Y = np.load(Y_train_path)
	earlystopping = keras.callback.EarlyStopping(monitor = "val_acc", min_delta = 0, patience = 0,
												mode = 'auto', restore_best_weights = True)
	history = model.fit(X, Y, , validation_split = 0.1, shuffle = True, 
						batch_size = 10, epochs = 20)
	training_history.append(history.history)
	model.save("twd_model.h5")
	return training_history

if __name__ == "__main__":
	Tx = 5511 # number of time steps input to the model from the spectrogram
	n_Freq = 101 # number of frequencies input to the model at each time step of the spectrogram
	Ty = 1375 # number of time steps in the output of our model
	model = model(input_shape = (Tx, n_Freq))
	print(model.summary())
	X_train_path = "" # file path to training samples X
	Y_train_path = "" # file path to training samples Y
	training_history = train_model(X_train_path, Y_train_path)
	history = [d['acc'] for d in training_history]
	history = sum(history, [])
	plt.plot(history)
	plt.xlabel("Number of Training Batches")
	plt.ylabel("Accuracy (%)")
