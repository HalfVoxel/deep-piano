from collections import namedtuple
import numpy as np
from keras import Sequential, Model
from keras.callbacks import ModelCheckpoint, LambdaCallback, TensorBoard
from keras.layers import LSTM, Dropout, Dense, Activation, Input, Embedding, Concatenate, BatchNormalization, GRU
from keras.utils import to_categorical
import read_data
import os
import sys
from read_data import Item
from fractions import Fraction
import subprocess
from datetime import datetime

# sys.path.insert(1, os.path.join(sys.path[0], 'music21'))
# print(sys.path)
from music21 import stream

import matplotlib
matplotlib.use('Agg')
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['CMU Serif']
rcParams['font.size'] = 16
#Direct input
#Options
rcParams['mathtext.fontset'] = 'cm'

Dataset = namedtuple("Dataset", ["input", "output"])
TrainingData = namedtuple("TrainingData", ["input", "output", "indices"])
sequence_length = 100


class IndexMapping:
    def __init__(self, values):
        self.index2value = values[:]
        self.value2index = {value: index for index, value in enumerate(self.index2value)}

    def __len__(self):
        return len(self.index2value)

    def to_index(self, value):
        return self.value2index[value]

    def to_value(self, index):
        return self.index2value[index]


def analyze_data(songs):
    # Concatenate all songs
    notes = []
    for i in range(len(songs)):
        song = songs[i] = [
            Item(
                note.pitches,
                Fraction(note.duration).limit_denominator(100),
                Fraction(note.beat).limit_denominator(100),
                Fraction(note.offset_to_next).limit_denominator(100)
            ) for note in songs[i]
        ]

        for note in song:
            # pitches duration beat offset_to_next
            notes.append(note)


    # get all pitch names
    pitches = IndexMapping(sorted(set([note.pitches for note in notes])))
    durations = IndexMapping(sorted(set([note.duration for note in notes])))
    offsets = IndexMapping(sorted(set([note.offset_to_next for note in notes])))
    beats = IndexMapping(sorted(set([note.beat for note in notes])))
    # print(len(durations), durations.index2value)
    # print(len(pitches), pitches.index2value)
    # print(len(offsets), offsets.index2value)
    # print(len(beats), beats.index2value)

    return pitches, durations, beats, offsets


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


def pitch(item):
    # 60 is just a made up multiplier to keep the data roughly around 1
    return (sum(item.pitches)/len(item.pitches)) / 60

def prepare_input(songs, sequence_length, pitches, durations, beats, offsets):
    network_input = []
    network_input2 = []
    network_input3 = []
    network_input4 = []
    network_input5 = []
    network_input6 = []
    network_output = []
    network_output2 = []
    network_output3 = []
    indices = []

    for song_index, notes in enumerate(songs):
        song_input = [pitches.to_index(char.pitches) for char in notes]
        song_input2 = [durations.to_index(char.duration) for char in notes]
        song_input3 = [offsets.to_index(char.offset_to_next) for char in notes]
        song_input4 = [beats.to_index(char.beat) for char in notes]
        song_input6 = [pitch(char) for char in notes]

        # create input sequences and the corresponding outputs
        for i in range(0, len(notes) - sequence_length, 1):
            start = i
            end = i + sequence_length
            network_input.append(song_input[start:end])
            network_input2.append(song_input2[start:end])
            network_input3.append(song_input3[start:end])
            network_input4.append(song_input4[start:end])
            network_input5.append(song_input4[end])
            network_input6.append(song_input6[start:end])
            network_output.append(song_input[end])
            network_output2.append(song_input2[end])
            network_output3.append(song_input3[end])
            indices.append(song_index)

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length))
    network_input2 = np.reshape(network_input2, (n_patterns, sequence_length))
    network_input3 = np.reshape(network_input3, (n_patterns, sequence_length))
    network_input4 = np.reshape(network_input4, (n_patterns, sequence_length))
    network_input6 = np.reshape(network_input6, (n_patterns, sequence_length, 1))
    indices = np.reshape(indices, (n_patterns,))

    network_input5 = to_categorical(network_input5, num_classes=len(beats))

    if np.max(network_input) >= len(pitches) or np.min(network_input) < 0:
        raise Exception("Invalid pitch index")

    if np.max(network_input2) >= len(durations) or np.min(network_input2) < 0:
        raise Exception("Invalid duration index")

    if np.max(network_input3) >= len(offsets) or np.min(network_input3) < 0:
        raise Exception("Invalid offset index")

    if np.max(network_input4) >= len(beats) or np.min(network_input4) < 0:
        raise Exception("Invalid beat index")

    # Hopefully few notes
    # network_input2 = to_categorical(network_input2)
    # assert network_input2.shape == (n_patterns, sequence_length, num_durations)
    # normalize input
    # network_input = network_input / num_notes

    network_output = to_categorical(network_output, num_classes=len(pitches))
    network_output2 = to_categorical(network_output2, num_classes=len(durations))
    network_output3 = to_categorical(network_output3, num_classes=len(offsets))

    perm = np.random.permutation(n_patterns)
    return TrainingData(
        [network_input[perm], network_input2[perm], network_input3[perm], network_input4[perm], network_input5[perm], network_input6[perm]],
        [network_output[perm], network_output2[perm], network_output3[perm]],
        indices[perm]
    )


def create_model(sequence_length, pitches, durations, beats, offsets):
    # 500*12 + (12+4+4+4+50)*512 + 512*512 + 512*512 + 512*512 + 512*(500+50+50+50)
    model = Sequential()

    in_pitch = Input(shape=(sequence_length,), name="in_pitches")
    in_duration = Input(shape=(sequence_length,), name="in_durations")
    in_offset = Input(shape=(sequence_length,), name="in_offsets")
    in_beat = Input(shape=(sequence_length,), name="in_beats")
    in_current_beat = Input(shape=(len(beats),), name="in_current_beat")
    in_pitch_float = Input(shape=(sequence_length,1), name="in_pitches_float")

    emb_pitch = Embedding(len(pitches), 12, name="pitch_embedding")(in_pitch)
    emb_duration = Embedding(len(durations), 4, name="duration_embedding")(in_duration)
    emb_offset = Embedding(len(offsets), 4, name="offset_embedding")(in_offset)
    emb_beat = Embedding(len(beats), 4, name="beat_embedding")(in_beat)

    x = Concatenate(axis=2)([emb_pitch, emb_duration, emb_offset, emb_beat, in_pitch_float])
    x = GRU(200, return_sequences=True, reset_after=True, recurrent_activation='sigmoid')(x)
    x = Dropout(0.5)(x)
    x = GRU(200, return_sequences=True, reset_after=True, recurrent_activation='sigmoid')(x)
    x = Dropout(0.5)(x)
    x = GRU(400, reset_after=True, recurrent_activation='sigmoid')(x)
    x = Concatenate(axis=1)([x, in_current_beat])
    x = Dense(400, activation="relu")(x)
    x = BatchNormalization()(x)
    drop = Dropout(0.5)(x)
    vocab = Dense(len(pitches))(drop)
    out_pitch = Activation('softmax', name="notes")(vocab)

    duration = Dense(len(durations))(drop)
    out_duration = Activation('softmax', name="durations")(duration)

    offset = Dense(len(offsets))(drop)
    out_offset = Activation('softmax', name="offsets")(offset)

    model = Model(inputs=[in_pitch, in_duration, in_offset, in_beat, in_current_beat, in_pitch_float], outputs=[out_pitch, out_duration, out_offset])
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model

id = datetime.now().strftime("%Y%m%d-%H%M") + "_" + subprocess.check_output("git rev-parse HEAD", shell=True).decode('utf-8')[0:6]
output_folder = "results/" + str(id)

def train_model():
    data, names = load_data("data/final_fantasy")
    print("Analyzing...")
    pitches, durations, beats, offsets = analyze_data(data)
    print("Preparing input arrays...")
    prepared = prepare_input(data, sequence_length, pitches, durations, beats, offsets)
    print("Creating model...")
    # train, test, validation = split_data(prepared)
    model = create_model(sequence_length, pitches, durations, beats, offsets)
    print(model.count_params())
    print(model.summary())
    os.makedirs("checkpoints", exist_ok=True)
    filepath = "checkpoints/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"

    checkpoint = ModelCheckpoint(
        filepath, monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )

    os.makedirs(output_folder)
    tensorboard = TensorBoard(log_dir='./logs/' + id + "/", histogram_freq=0, write_graph=True, write_images=False)
    callbacks_list = [checkpoint, tensorboard, LambdaCallback(on_epoch_begin=lambda epoch, logs: generate(model, prepared, epoch, pitches, durations, beats, offsets, names))]
    # model.load_weights('checkpoints/weights.hdf5')
    model.fit(prepared.input, prepared.output, epochs=50, batch_size=1000, callbacks=callbacks_list, validation_split=0.05)


def load_and_generate():
    data, names = load_data("data/final_fantasy")
    print("Analyzing...")
    pitches, durations, beats, offsets = analyze_data(data)
    print("Preparing input arrays...")
    prepared = prepare_input(data, sequence_length, pitches, durations, beats, offsets)
    print("Creating model...")
    model = create_model(sequence_length, pitches, durations, beats, offsets)
    model.load_weights("<todo>")

    for i in range(10):
        generate(model, prepared, i, pitches, durations, beats, offsets, names)


def generate(model, data, epoch, pitches, durations, beats, offsets, song_names):
    print("Generating some stuff")
    # create a sequence of note/chord predictions
    start = np.random.randint(0, data.input[0].shape[0] - 1)
    print("Based on " + song_names[data.indices[start]])

    input_pitch, input_duration, input_offset, input_beat, input_pitch_float = list(data.input[0][start]), list(data.input[1][start]), list(data.input[2][start]), list(data.input[3][start]), list(data.input[5][start])
    sequence_length = len(input_pitch)

    # input_pitch = list(np.random.random_integers(0, len(pitches)-1, size=sequence_length))
    # input_duration = list(np.random.random_integers(0, len(durations)-1, size=sequence_length))
    # input_offset = list(np.random.random_integers(0, len(offsets)-1, size=sequence_length))
    # input_pitch_float = list(np.random.uniform(size=sequence_length))

    prediction_output = []

    # for i in range(sequence_length-1):
    #     print(beats.to_value(input_beat[i+1]), beats.to_value(input_beat[i]), offsets.to_value(input_offset[i]), (beats.to_value(input_beat[i]) + offsets.to_value(input_offset[i])) % 4)

    predictions = []
    # generate 500 input_pitch
    for note_index in range(500):
        prediction_input1 = np.reshape(input_pitch[note_index:note_index+sequence_length], (1, sequence_length))
        prediction_input2 = np.reshape(input_duration[note_index:note_index+sequence_length], (1, sequence_length))
        prediction_input3 = np.reshape(input_offset[note_index:note_index+sequence_length], (1, sequence_length))
        prediction_input4 = np.reshape(input_beat[note_index:note_index+sequence_length], (1, sequence_length))
        prediction_input6 = np.reshape(input_pitch_float[note_index:note_index+sequence_length], (1, sequence_length, 1))
        new_beat = Fraction(beats.to_value(input_beat[-1]) + offsets.to_value(input_offset[-1])) % 4

        try:
            new_beat_index = beats.to_index(new_beat)
        except:
            print("New beat didn't exist, resetting to beat zero")
            new_beat_index = beats.to_index(0)

        prediction_input5 = to_categorical(np.array([new_beat_index]), num_classes=len(beats))

        prediction_note, prediction_duration, prediction_offset = model.predict([prediction_input1, prediction_input2, prediction_input3, prediction_input4, prediction_input5, prediction_input6], verbose=0)
        index1 = np.random.choice(len(prediction_note[0]), p=prediction_note[0]**2 / np.sum(prediction_note[0]**2))
        index2 = np.random.choice(len(prediction_duration[0]), p=prediction_duration[0]**2 / np.sum(prediction_duration[0]**2))
        index3 = np.random.choice(len(prediction_offset[0]), p=prediction_offset[0]**2 / np.sum(prediction_offset[0]**2))
        predictions.append([prediction_note[0], prediction_duration[0], prediction_offset[0]])
        result = Item(pitches.to_value(index1), durations.to_value(index2), new_beat, offsets.to_value(index3))

        prediction_output.append(result)
        input_pitch.append(index1)
        input_duration.append(index2)
        input_offset.append(index3)
        input_beat.append(new_beat_index)
        input_pitch_float.append(pitch(result))

    for i in range(3):
        for x in predictions:
            x[i] /= np.max(x[i])

        ps = np.array([x[i] for x in predictions])
        ps = ps.transpose()
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.imshow(ps, interpolation="nearest", origin='lower', cmap=plt.get_cmap("inferno"))
        ax.set_ylabel(["Pitch", "Duration", "Offset"][i])
        ax.set_xlabel("Time (note index)")
        pdf = PdfPages(output_folder + "/epoch_"+str(epoch) + '_' + ["notes", "durations", "offsets"][i] + '.pdf')
        pdf.savefig(fig)
        pdf.close()
        f = open(output_folder + "/epoch_" +str(epoch) + ".txt", "w")
        f.write(song_names[data.indices[start]])
        f.close()

    output_notes = read_data.convert_to_notes(prediction_output)
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=output_folder + '/epoch_{}.mid'.format(epoch))


load_and_generate()
# train_model()
