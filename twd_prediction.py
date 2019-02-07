import numpy as np
from pydub import AudioSegment
import random
import sys
import io
import os
import glob
from td_utils import *
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam

def detect_triggerword(filename):
	plt.subplot(2, 1, 1)

	x = graph_spectrogram(filename)
	x = np.transpose(x)
	x = np.expand_dims(x, axis = 0)
	
	predictions = model.predict(x)
	new_prediction = predictions[0, :, 0]>0.5
	
	plt.subplot(2, 1, 2)
	plt.plot(new_prediction)
	plt.ylabel('probability')
	plt.show()
	return predictions

if __name__ == "main":
	filename = '' # file path to wav file to predict if trigger word exists
	prediction = detect_triggerword(filename)