from collections import namedtuple
import numpy as np
from keras import Sequential, Model
from keras.callbacks import ModelCheckpoint, LambdaCallback, TensorBoard
from keras.layers import LSTM, Dropout, Dense, Activation, Input, Embedding, Concatenate
from keras.utils import to_categorical
import read_data
import os
import sys
# sys.path.insert(1, os.path.join(sys.path[0], 'music21'))
# print(sys.path)
from music21 import note, chord, instrument, stream

Dataset = namedtuple("Dataset", ["input", "output"])
TrainingData = namedtuple("TrainingData", ["input", "output"])
sequence_length = 100


def analyze_data(songs):
    # Concatenate all songs
    notes = sum(songs, [])
    # get all pitch names
    notes_names = sorted(set([note.split(":")[0] for note in notes]))
    duration_names = sorted(set([str(round(float(note.split(":")[1]),4)) for note in notes]))

    # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(notes_names))
    duration_to_int = dict((dur, number) for number, dur in enumerate(duration_names))
    int_to_note = {value: key for key, value in note_to_int.items()}
    int_to_duration = {value: key for key, value in duration_to_int.items()}
    return lambda x: note_to_int[x.split(":")[0]], lambda x: duration_to_int[str(round(float(x.split(":")[1]), 4))], len(note_to_int), len(duration_to_int), int_to_note, int_to_duration


def load_data(midi_path):
    ''' Returns list of note squences '''
    return read_data.get_pickle()


def split_data(data):
    ''' Splits data into train, test and validation datasets '''
    # Cumulative split fractions
    train_split = 0.80
    test_split = 0.90
    validation_split = 1.0

    train_i = int(train_split * len(data.input))
    test_i = int(test_split * len(data.input))
    validation_i = int(validation_split * len(data.input))

    train = TrainingData(data.input[0:train_i], data.output[0:train_i])
    test = TrainingData(data.input[train_i:test_i], data.output[train_i:test_i])
    validation = TrainingData(data.input[test_i:validation_i], data.output[test_i:validation_i])
    return train, test, validation


def prepare_input(songs, num_notes, num_durations, note_to_int, duration_to_int, sequence_length):

    network_input = []
    network_input2 = []
    network_output = []
    network_output2 = []
    print(num_notes, num_durations)

    for notes in songs:
        # create input sequences and the corresponding outputs
        for i in range(0, len(notes) - sequence_length, 1):
            sequence_in = notes[i:i + sequence_length]
            sequence_out = notes[i + sequence_length]
            network_input.append([note_to_int(char) for char in sequence_in])
            network_input2.append([duration_to_int(char) for char in sequence_in])
            network_output.append(note_to_int(sequence_out))
            network_output2.append(duration_to_int(sequence_out))

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length))
    network_input2 = np.reshape(network_input2, (n_patterns, sequence_length))

    if np.max(network_input2) >= num_durations or np.min(network_input2) < 0:
        raise Exception("Invalid duration index")

    # Hopefully few notes
    # network_input2 = to_categorical(network_input2)
    # assert network_input2.shape == (n_patterns, sequence_length, num_durations)
    # normalize input
    # network_input = network_input / num_notes

    network_output = to_categorical(network_output, num_classes=num_notes)
    network_output2 = to_categorical(network_output2, num_classes=num_durations)

    perm = np.random.permutation(n_patterns)
    return TrainingData([network_input[perm], network_input2[perm]], [network_output[perm], network_output2[perm]])


def create_model(sequence_length, n_vocab, n_durations):
    model = Sequential()

    in_note = Input(shape=(sequence_length,))
    in_duration = Input(shape=(sequence_length,))
    en = Embedding(n_vocab,4, name="note_embedding")(in_note)
    ed = Embedding(n_durations,4, name="duration_embedding")(in_duration)
    x = Concatenate(axis=2)([en,ed])
    x = LSTM(512, return_sequences=True, activation="sigmoid")(x)
    x = Dropout(0.3)(x)
    x = LSTM(256, return_sequences=True, activation="sigmoid")(x)
    x = Dropout(0.3)(x)
    x = LSTM(512, activation="sigmoid")(x)
    x = Dense(256, activation="relu")(x)
    drop = Dropout(0.3)(x)
    vocab = Dense(n_vocab)(drop)
    out_note = Activation('softmax', name="notes")(vocab)

    duration = Dense(n_durations)(drop)
    out_duration = Activation('softmax', name="durations")(duration)

    model = Model(inputs=[in_note, in_duration], outputs=[out_note, out_duration])
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model


def create_comp_model(sequence_length, n_vocab, n_durations):
    model = Sequential()

    in_note = Input(shape=(sequence_length,))
    in_duration = Input(shape=(sequence_length,))
    in_future_note = Input(shape=(sequence_length,))
    in_future_duration = Input(shape=(sequence_length,))

    for note, dur in [(in_note, in_duration), (in_future_note, in_future_duration)]:
        en = Embedding(n_vocab,4)(note)
        ed = Embedding(n_durations,4)(dur)
        x = Concatenate(axis=2)([en,ed])
        x = Bidirectional(LSTM(512, return_sequences=True))(x)
        x = Dropout(0.3)(x)
        x = LSTM(256, return_sequences=True)(x)
        x = Dropout(0.3)(x)
        x = LSTM(512)(x)

    en2 = Embedding(n_vocab,4, name="note_embedding2")(in_future_note)
    ed2 = Embedding(n_durations,4, name="duration_embedding2")(in_future_duration)
    x2 = Concatenate(axis=2)([en2,ed2])
    x2 = Bidirectional(LSTM(512, return_sequences=True))(x2)

    x = Concatenate(axis=2)([x,x2])

    
    x = Dense(256)(x)
    drop = Dropout(0.3)(x)
    vocab = Dense(n_vocab)(drop)
    out_note = Activation('softmax', name="notes")(vocab)

    duration = Dense(n_durations)(drop)
    out_duration = Activation('softmax', name="durations")(duration)

    model = Model(inputs=[in_note, in_duration], outputs=[out_note, out_duration])
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model


def train_model():
    data = load_data("data/final_fantasy")
    note_to_int, duration_to_int, num_notes, num_durations, int_to_note, int_to_duration = analyze_data(data)
    prepared = prepare_input(data, num_notes, num_durations, note_to_int, duration_to_int, sequence_length)
    # train, test, validation = split_data(prepared)
    model = create_model(sequence_length, num_notes, num_durations)
    os.makedirs("checkpoints", exist_ok=True)
    filepath = "checkpoints/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"

    checkpoint = ModelCheckpoint(
        filepath, monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
    callbacks_list = [checkpoint, tensorboard, LambdaCallback(on_epoch_begin=lambda epoch, logs: generate(model, prepared, num_notes, num_durations, int_to_note, int_to_duration, epoch))]
    # model.load_weights('checkpoints/weights.hdf5')
    model.fit(prepared.input, prepared.output, epochs=200, batch_size=200, callbacks=callbacks_list, validation_split=0.05)


def load_and_generate():
    # model = create_model()
    # # Load the weights to each node
    # model.load_weights('checkpoints/weights.hdf5')
    pass


def generate(model, data, num_notes, num_durations, int_to_note, int_to_duration, epoch):
    print("Generating some stuff")
    # create a sequence of note/chord predictions
    start = np.random.randint(0, data.input[0].shape[0] - 1)

    notes, durations = list(data.input[0][start]), list(data.input[1][start])
    prediction_output = []
    # generate 500 notes
    for note_index in range(500):
        prediction_input1 = np.reshape(notes, (1, len(notes)))
        prediction_input2 = np.reshape(durations, (1, len(durations)))
        prediction_note, prediction_duration = model.predict([prediction_input1, prediction_input2], verbose=0)
        index1 = np.argmax(prediction_note)
        index2 = np.argmax(prediction_duration)
        result = (int_to_note[index1], int_to_duration[index2])
        prediction_output.append(result[0]+":"+result[1])
        notes.append(index1)
        durations.append(index2)
        notes = notes[1:]
        durations = durations[1:]

    output_notes = read_data.convert_to_notes(prediction_output)
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='output_epoch_{}.mid'.format(epoch))


train_model()
