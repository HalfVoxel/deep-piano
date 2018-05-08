import music21
from music21 import *
import glob
import pickle
from pickle import load, dump

def get_notes():
    """ Get all the notes and chords from the midi files in the data directory """
    notes = []

    for file in glob.glob('data/bach/*/*.mid'): #only reads Bach
        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    with open('data/notes/notes.pickle', 'wb') as f:
        pickle.dump(notes, f)

    return notes

def get_pickle():

    with open('data/notes/notes.pickle', 'rb') as f:
        unpickled = load(f)

    return unpickled
