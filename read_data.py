import music21
from music21 import *
import glob
import pickle
from pickle import load, dump

def get_notes():
    """ Get all the notes and chords from the midi files in the data directory """
    notes = []

<<<<<<< HEAD:read-data.py
    for file in glob.glob('data/bach/*/*.mid'): #only reads Bach
        midi = converter.parse(file)

=======
    for file in glob.glob('data/*/*.mid')[0:4]:
>>>>>>> aa8c18e1ac959e0bbfcb18e454df459efca93c21:read_data.py
        print("Parsing %s" % file)
        midi = converter.parse(file)

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

<<<<<<< HEAD:read-data.py
    with open('data/notes/notes.pickle', 'wb') as f:
        pickle.dump(notes, f)

    return notes

def get_pickle():

    with open('data/notes/notes.pickle', 'rb') as f:
        unpickled = load(f)

    return unpickled
=======
    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes

if __name__ == "__main__":
    x = get_notes()
    print(x)
>>>>>>> aa8c18e1ac959e0bbfcb18e454df459efca93c21:read_data.py
