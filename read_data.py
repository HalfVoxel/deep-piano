import music21
from music21 import *
import glob
import pickle
from pickle import load, dump
from recordclass import recordclass


# Chord = recordclass("Chord", "start end notes")
# def merge_notes(notes):
#     result = []
#     for note in notes:
#         if len(result) > 0 and result[-1].start == note.start and result[-1].end == note.end:
#             if isinstance(result[-1], Chord):
#                 result[-1].notes.append(note)
#             else:
#                 result[-1] = Chord(note.start, note.end, [result[-1], note])
#         else:
#             result.append(note)


#     return result


# def quantize(midi, tracks):
#     quanta = int(midi.time_to_tick(60/midi.get_tempo_changes()[1][0])/6)  # min(durations)
#     durations = []
#     for _,_,track in tracks:
#         for note in track:
#             note.start = midi.tick_to_time(quanta*int(round(midi.time_to_tick(note.start)/quanta)))
#             # note.end = midi.tick_to_time(quanta*int(round(midi.time_to_tick(note.end)/quanta)))
#             duration = midi.time_to_tick((note.end - note.start)/quanta)
#             note.end = note.start + midi.tick_to_time(duration) * quanta
#             print(note.start/quanta)
#             if isinstance(note, Chord):
#                 for n in note.notes:
#                     n.start = note.start
#                     n.end = note.end
#             durations.append(midi.time_to_tick(note.start))

#     durations = [x/quanta for x in durations]
#     for x in durations:
#         if abs(x - round(x)) > 0.1:
#             print("Failed to quantize: " + str(x))
#             print(durations)
#             exit(1)

Item = recordclass("Item", "pitches duration beat offset_to_next")


def pitch(e):
    if isinstance(e, note.Note):
        return e.pitch.ps
    else:
        return sum(x.ps for x in e.pitches)/len(e.pitches)


def read_midi(file):
    midi = converter.parse(file)
    tsFourFour = meter.TimeSignature('4/4')
    if not midi.elements[0].timeSignature.ratioEqual(tsFourFour):
        print("Skipped because time signature was " + str(midi.elements[0].timeSignature))
        return []

    notes_to_parse = midi.flat.notes
    notes_to_parse = [x for x in notes_to_parse if not isinstance(x, note.Rest) and not isinstance(x, tempo.MetronomeMark)]
    notes_to_parse.sort(key=lambda x: (x.offset, pitch(x)))

    file_notes = []
    for element in notes_to_parse:
        pitches = []
        if isinstance(element, note.Note):
            # Note rounding is done of the pitches
            # On a piano every key is an integer, so this is reasonable.
            # We don't really care about microtones.
            # Note that this is still a tuple.
            pitches = (int(element.pitch.ps),)
        elif isinstance(element, chord.Chord):
            pitches = [int(x.ps) for x in element.pitches]
            # Make sure the pitches come in a canonical order
            pitches.sort()
            # Make it hashable
            pitches = tuple(pitches)
        else:
            assert False

        file_notes.append(Item(pitches, element.duration.quarterLength, element.beat % 4, element.offset))

    # Convert raw offsets to delta offsets
    for i in range(len(file_notes)-1):
        item = file_notes[i]
        item.offset_to_next = file_notes[i+1].offset_to_next - item.offset_to_next

    file_notes[-1].offset_to_next = 0
    return file_notes


def get_notes():
    """ Get all the notes and chords from the midi files in the data directory """
    notes = []
    # for file in glob.glob('data/final_fantasy/*.mid'):
    for file in glob.glob('data/bach/*/*.mid'):  # only reads Bach
        print("Parsing %s" % file)
        notes.append(read_midi(file))

    with open('data/notes/notes.pickle', 'wb') as f:
        pickle.dump(notes, f)

    # file = "data/final_fantasy/ahead_on_our_way_piano.mid"
    # file = "data/bach/cantatas/jesu1.mid"
    # file = "data/bach/sinfon/sinfon1.mid"
    # # print(read_midi(file))
    # midi_stream = stream.Stream(convert_to_notes(read_midi(file)))
    # midi_stream.write('midi', fp='all.mid')

    return notes


def get_pickle():

    with open('data/notes/notes.pickle', 'rb') as f:
        unpickled = load(f)

    return unpickled


def convert_to_notes(input_notes):
    # create stream from predictions
    offset = 0
    output_notes = []
    # create note and chord objects based on the values generated by the model
    for item in input_notes:
        if len(item.pitches) > 1:
            # pattern is a chord
            notes = []
            for pitch in item.pitches:
                new_note = note.Note(ps=pitch)
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_element = chord.Chord(notes)
        else:
            # pattern is a note
            new_element = note.Note(ps=item.pitches[0])
            new_element.storedInstrument = instrument.Piano()

        new_element.duration.quarterLength = item.duration
        new_element.offset = offset
        offset += item.offset_to_next
        output_notes.append(new_element)
    return output_notes


if __name__ == "__main__":
    x = get_notes()
    # print(x)
