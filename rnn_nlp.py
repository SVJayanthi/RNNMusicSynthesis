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
from decode import *

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

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

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
    
    print(idxenc.size)
    print(idxenc.shape)
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
    """
    dataset = sequences.map(lambda x: (x[:-1], x[1:]))
    
    """
    for input_example, target_example in  dataset.take(1):
        print ('Input data: ', input_example.numpy())
        print ('Target data:', target_example.numpy())
    """
    # Batch size
    BATCH_SIZE = 17
    
    # Buffer size to shuffle the dataset
    BUFFER_SIZE = 10000
    
    dataset = dataset.shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=False)
    print(dataset)
    
    """
    for input_example, target_example in  dataset.take(1):
        print('enter loop')
        print ('Input data: ', input_example.numpy())
        print ('Target data:', target_example.numpy())
    """
    
    # The embedding dimension
    embedding_dim = 256
    
    # Number of RNN units
    rnn_units = 1024
    
    model = build_model(vocab_size = len(vocab.itos), embedding_dim=embedding_dim,
                        rnn_units=rnn_units, batch_size=BATCH_SIZE)    
    
    print(model.summary())
    
    example_batch_predictions = None
    input_batch = None
    target_batch = None
    for input_example_batch, target_example_batch in dataset.take(1):
        example_batch_predictions = model(input_example_batch)
        input_batch = input_example_batch
        target_batch = target_example_batch
        print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
    

    sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
    sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
    
    print(idxenc2npenc(input_batch[0].numpy(), vocab))
    print("predicted values")
    print(idxenc2npenc(sampled_indices, vocab))
    
    example_batch_loss  = loss(target_batch, example_batch_predictions)
    print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
    print("scalar_loss:      ", example_batch_loss.numpy().mean())
    
    model.compile(optimizer='adam', loss=loss)
    
    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    
    checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)
    
    EPOCHS=10
    
    history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
    
    model = build_model(len(vocab.itos), embedding_dim, rnn_units, batch_size=1)

    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    
    model.build(tf.TensorShape([1, None]))
    
    print(model.summary()) 
    
    # Evaluation step (generating text using the learned model)

    # Number of characters to generate
    num_generate = 1000
      
    # Converting our start string to numbers (vectorizing)
    input_eval = idxenc[:10]
    input_eval = tf.expand_dims(input_eval, 0)
      
    # Empty string to store our results
    music_generated = []
      
    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 1.0
      
    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)
      
        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
      
        # We pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)
      
        print(predicted_id)
        print(len(vocab.itos))
        print(idxenc2npenc(np.array([predicted_id]), vocab))
        music_generated.append(predicted_id)
      
    print(music_generated)
    
    generated_stream = idxenc2stream(music_generated, vocab)
    generated_stream.write('midi', 'generated/synthesized_song.mid')
