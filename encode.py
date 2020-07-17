# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 22:51:44 2020

@author: srava
"""


import music21
import numpy as np
from vocab import *

def file2idxenc(path, vocab):
    "Converts midi files to index encodings"
    stream = file_stream(path)
    chordarr = stream_chordarr(stream)
    npenc = chordarr_npenc(chordarr)
    return npenc2idxenc(npenc, vocab)

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
