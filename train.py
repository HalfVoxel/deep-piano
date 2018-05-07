from collections import namedtuple
import numpy as np
from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM, Dropout, Dense, Activation
from keras.utils import to_categorical
import read_data
import os

Dataset = namedtuple("Dataset", ["input", "output"])
TrainingData = namedtuple("TrainingData", ["input", "output"])
sequence_length = 100

def analyze_data(notes):
	# get all pitch names
	pitchnames = sorted(set(notes))

	# create a dictionary to map pitches to integers
	note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
	return note_to_int


def load_data(midi_path):
	''' Returns list of note squences '''
	return read_data.get_notes()

def split_data(data):
	''' Splits data into train, test and validation datasets '''
	# Cumulative split fractions
	train_split = 0.80
	test_split = 0.90
	validation_split = 1.0

	train_i = int(train_split*len(data.input))
	test_i = int(test_split*len(data.input))
	validation_i = int(validation_split*len(data.input))

	train = TrainingData(data.input[0:train_i], data.output[0:train_i])
	test = TrainingData(data.input[train_i:test_i], data.output[train_i:test_i])
	validation = TrainingData(data.input[test_i:validation_i], data.output[test_i:validation_i])
	return train, test, validation

def prepare_input(notes, note_to_int):

	network_input = []
	network_output = []
	print(len(notes))

	# create input sequences and the corresponding outputs
	for i in range(0, len(notes) - sequence_length, 1):
		sequence_in = notes[i:i + sequence_length]
		sequence_out = notes[i + sequence_length]
		network_input.append([note_to_int[char] for char in sequence_in])
		network_output.append(note_to_int[sequence_out])

	n_patterns = len(network_input)

	# reshape the input into a format compatible with LSTM layers
	network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
	# normalize input
	network_input = network_input / len(note_to_int)

	network_output = to_categorical(network_output)
	return TrainingData(network_input, network_output)


def create_model(input_shape, n_vocab):
	model = Sequential()
	print(input_shape)
	assert len(input_shape) == 3
	assert input_shape[2] == 1

	model.add(LSTM(
		512,
		input_shape=(input_shape[1], input_shape[2]),
		return_sequences=True
	))
	model.add(Dropout(0.3))
	model.add(LSTM(512, return_sequences=True))
	model.add(Dropout(0.3))
	model.add(LSTM(512))
	model.add(Dense(256))
	model.add(Dropout(0.3))
	model.add(Dense(n_vocab))
	model.add(Activation('softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
	return model

def train_model():
	data = load_data("data/final_fantasy")
	note_to_int = analyze_data(data)
	prepared = prepare_input(data, note_to_int)
	train, test, validation = split_data(prepared)
	model = create_model(train.input.shape, len(note_to_int))
	os.makedirs("checkpoints", exist_ok=True)
	filepath = "checkpoints/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"    

	checkpoint = ModelCheckpoint(
	    filepath, monitor='loss', 
	    verbose=0,
	    save_best_only=True,
	    mode='min'
	)
	callbacks_list = [checkpoint]

	model.fit(train.input, train.output, epochs=200, batch_size=200, callbacks=callbacks_list)

def generate():
	model = create_model()
	# Load the weights to each node
	model.load_weights('checkpoints/weights.hdf5')

train_model()