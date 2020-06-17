# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 19:43:43 2020

@author: srava
"""

import tensorflow as tf

import numpy as np
import os
import time

import music21

from vocab import MusicVocab

BPB = 4 # beats per bar
TIMESIG = f'{BPB}/4' # default time signature
PIANO_RANGE = (21, 108)
VALTSEP = -1 # separator value for numpy encoding
VALTCONT = -2 # numpy value for TCONT - needed for compressing chord array

SAMPLE_FREQ = 4
NOTE_SIZE = 128
DUR_SIZE = (10*BPB*SAMPLE_FREQ)+1 # Max length - 8 bars. Or 16 beats/quarternotes
MAX_NOTE_DUR = (8*BPB*SAMPLE_FREQ)


#File to stream
def file_stream(path):
    if isinstance(path, music21.midi.MidiFile): return music21.midi.translate.midiFileToStream(path)
    return music21.converter.parse(path)

#Stream to array
def stream_chordarr(stream, note_size=NOTE_SIZE, sample_freq=SAMPLE_FREQ, max_note_dur=MAX_NOTE_DUR):
    """
    4/4 time
    note * instrument * pitch
    """
    
    highest_time = max(stream.flat.getElementsByClass('Note').highestTime, stream.flat.getElementsByClass('Chord').highestTime)
    maxTimeStep = round(highest_time * sample_freq)+1
    score_arr = np.zeros((maxTimeStep, len(stream.parts), NOTE_SIZE))

    def note_data(pitch, note):
        return (pitch.midi, int(round(note.offset*sample_freq)), int(round(note.duration.quarterLength*sample_freq)))

    for idx,part in enumerate(stream.parts):
        notes=[]
        for elem in part.flat:
            if isinstance(elem, music21.note.Note):
                notes.append(note_data(elem.pitch, elem))
            if isinstance(elem, music21.chord.Chord):
                for p in elem.pitches:
                    notes.append(note_data(p, elem))
                
        # sort notes by offset (1), duration (2) so that hits are not overwritten and longer notes have priority
        notes_sorted = sorted(notes, key=lambda x: (x[1], x[2])) 
        for n in notes_sorted:
            if n is None: continue
            pitch,offset,duration = n
            if max_note_dur is not None and duration > max_note_dur: duration = max_note_dur
            score_arr[offset, idx, pitch] = duration
            score_arr[offset+1:offset+duration, idx, pitch] = VALTCONT      # Continue holding note
    return score_arr


# Convert array to NP Encoding
def chordarr_npenc(chordarr, skip_last_rest=True):
    # combine instruments
    result = []
    wait_count = 0
    for idx,timestep in enumerate(chordarr):
        flat_time = timestep2npenc(timestep)
        if len(flat_time) == 0:
            wait_count += 1
        else:
            # pitch, octave, duration, instrument
            if wait_count > 0: result.append([VALTSEP, wait_count])
            result.extend(flat_time)
            wait_count = 1
    if wait_count > 0 and not skip_last_rest: result.append([VALTSEP, wait_count])
    return np.array(result, dtype=int).reshape(-1, 2) # reshaping. Just in case result is empty

# Note: not worrying about overlaps - as notes will still play. just look tied
def timestep2npenc(timestep, note_range=PIANO_RANGE, enc_type=None):
    # inst x pitch
    notes = []
    for i,n in zip(*timestep.nonzero()):
        d = timestep[i,n]
        if d < 0: continue # only supporting short duration encoding for now
        if n < note_range[0] or n >= note_range[1]: continue # must be within midi range
        notes.append([n,d,i])
        
    notes = sorted(notes, key=lambda x: x[0], reverse=True) # sort by note (highest to lowest)
    
    if enc_type is None: 
        # note, duration
        return [n[:2] for n in notes] 
    if enc_type == 'parts':
        # note, duration, part
        return [n for n in notes]
    if enc_type == 'full':
        # note_class, duration, octave, instrument
        return [[n%12, d, n//12, i] for n,d,i in notes] 
    
        
# Convering np encodings into Tensors for use in model
    # single stream instead of note,dur
def npenc2idxenc(t, vocab, add_eos=False):
    "Transforms numpy array from 2 column (note, duration) matrix to a single column"
    "[[n1, d1], [n2, d2], ...] -> [n1, d1, n2, d2]"
    if isinstance(t, (list, tuple)) and len(t) == 2: 
        return [npenc2idxenc(x, vocab) for x in t]
    t = t.copy()
    
    t[:, 0] = t[:, 0] + vocab.note_range[0]
    t[:, 1] = t[:, 1] + vocab.dur_range[0]

    prefix = np.array([vocab.bos_idx, vocab.pad_idx])
    suffix = np.array([vocab.eos_idx]) if add_eos else np.empty(0, dtype=int)
    return np.concatenate([prefix, t.reshape(-1), suffix])
    
def position_enc(idxenc, vocab):
    "Calculates positional beat encoding."
    sep_idxs = (idxenc == vocab.sep_idx).nonzero()[0]
    sep_idxs = sep_idxs[sep_idxs+2 < idxenc.shape[0]] # remove any indexes right before out of bounds (sep_idx+2)
    dur_vals = idxenc[sep_idxs+1]
    dur_vals[dur_vals == vocab.mask_idx] = vocab.dur_range[0] # make sure masked durations are 0
    dur_vals -= vocab.dur_range[0]
    
    posenc = np.zeros_like(idxenc)
    posenc[sep_idxs+2] = dur_vals
    return posenc.cumsum()


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

if __name__ == '__main__':
    midi_file = "bwv772.mid"
    stream = file_stream(midi_file)
    chordarr = stream_chordarr(stream)
    npenc = chordarr_npenc(chordarr)
    #print(npenc)
    #print(len(npenc))
    vocab = MusicVocab.create()
    #print(vocab.stoi)
    idxenc = npenc2idxenc(npenc, vocab)
    #print(idxenc)
    """
    a = np.asarray(idxenc)
    np.savetxt("idxenc.csv", a, delimiter=",")
    """
    
    # The maximum length sentence we want for a single input in characters
    seq_length = 100
    examples_per_epoch = len(idxenc)//(seq_length+1)
    
    # Create training examples / targets
    char_dataset = tf.data.Dataset.from_tensor_slices(idxenc)
    sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
    """
    for i in char_dataset.take(15):
      print(vocab.itos[i.numpy()])
    for item in sequences.take(1):
        print(item.numpy())
        for i in item.numpy():
            print(vocab.itos[i])
      #print(repr(''.join(vocab.itos[item.numpy()])))
    
    for input_example, target_example in  dataset.take(1):
        print ('Input data: ', input_example.numpy())
        print ('Target data:', target_example.numpy())
"""
    dataset = sequences.map(split_input_target)
    
    # Batch size
    BATCH_SIZE = 64
    
    # Buffer size to shuffle the dataset
    BUFFER_SIZE = 10000
    
    dataset = dataset.shuffle(BUFFER_SIZE)
    
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=False).filter(lambda features, labels: tf.equal(tf.shape(labels)[0], BATCH_SIZE))
    print(dataset)
    
    for input_example, target_example in  dataset.take(1):
        print ('Input data: ', input_example.numpy())
        print ('Target data:', target_example.numpy())
        
    # The embedding dimension
    embedding_dim = 256
    
    # Number of RNN units
    rnn_units = 1024
    
    model = build_model(vocab_size = len(vocab.itos), embedding_dim=embedding_dim,
                        rnn_units=rnn_units, batch_size=BATCH_SIZE)    
    
    for input_example_batch, target_example_batch in dataset.take(1):
      example_batch_predictions = model(input_example_batch)
      print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
        
    print(model.summary())
    
    
    #sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
    #sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
