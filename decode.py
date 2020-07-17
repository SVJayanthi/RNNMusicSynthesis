# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 14:27:19 2020

@author: srava
"""

import music21
import numpy as np
from vocab import *


def idxenc2stream(arr, vocab, bpm=120):
    "Converts index encoding to music21 stream"
    npenc = idxenc2npenc(np.array(arr), vocab)
    return npenc2stream(npenc, bpm=bpm)

def idxenc2npenc(t, vocab, validate=True):
    if validate: t = to_valid_idxenc(t, vocab.npenc_range)
    t = t.copy().reshape(-1, 2)
    if t.shape[0] == 0: return t
        
    t[:, 0] = t[:, 0] - vocab.note_range[0]
    t[:, 1] = t[:, 1] - vocab.dur_range[0]
    
    if validate: return to_valid_npenc(t)
    return t

def to_valid_idxenc(t, valid_range):
    r = valid_range
    t = t[np.where((t >= r[0]) & (t < r[1]))]
    if t.shape[-1] % 2 == 1: t = t[..., :-1]
    return t

def to_valid_npenc(t):
    t = t[np.where((t[:, 0] >= VALTSEP) & (t[:, 0] < NOTE_SIZE) & (t[:, 1] >= 0))]
    
    return t
"""
def to_valid_npenc(t):
    is_note = (t[:, 0] < VALTSEP) | (t[:, 0] >= NOTE_SIZE)
    invalid_note_idx = is_note.argmax()
    invalid_dur_idx = (t[:, 1] < 0).argmax()

    invalid_idx = max(invalid_dur_idx, invalid_note_idx)
    if invalid_idx > 0: 
        if invalid_note_idx > 0 and invalid_dur_idx > 0: invalid_idx = min(invalid_dur_idx, invalid_note_idx)
        print('Non midi note detected. Only returning valid portion. Index, seed', invalid_idx, t.shape)
        return t[:invalid_idx]
    return t
"""

# Decoding process
# 1. NoteEnc -> numpy chord array
# 2. numpy array -> music21.Stream
def npenc2stream(arr, bpm=120):
    "Converts numpy encoding to music21 stream"
    chordarr = npenc2chordarr(np.array(arr)) # 1.
    return chordarr2stream(chordarr, bpm=bpm) # 2.

def stream2file(stream):
    if isinstance(stream, music21.stream.Stream): return music21.midi.translate.streamToMidiFile(stream)

##### DECODING #####

# 1.
def npenc2chordarr(npenc, note_size=NOTE_SIZE):
    num_instruments = 1 if len(npenc.shape) <= 2 else npenc.max(axis=0)[-1]
    
    max_len = npenc_len(npenc)
    # score_arr = (steps, inst, note)
    score_arr = np.zeros((max_len, num_instruments, note_size))
    
    idx = 0
    for step in npenc:
        n,d,i = (step.tolist()+[0])[:3] # or n,d,i
        if n < VALTSEP: continue # special token
        if n == VALTSEP:
            idx += d
            continue
        score_arr[idx,i,n] = d
    return score_arr

def npenc_len(npenc):
    duration = 0
    for t in npenc:
        if t[0] == VALTSEP: duration += t[1]
    return duration + 1


# 2.
def chordarr2stream(arr, sample_freq=SAMPLE_FREQ, bpm=120):
    duration = music21.duration.Duration(1. / sample_freq)
    stream = music21.stream.Score()
    stream.append(music21.meter.TimeSignature(TIMESIG))
    stream.append(music21.tempo.MetronomeMark(number=bpm))
    stream.append(music21.key.KeySignature(0))
    for inst in range(arr.shape[1]):
        p = partarr2stream(arr[:,inst,:], duration)
        stream.append(p)
    stream = stream.transpose(0)
    return stream

# 2b.
def partarr2stream(partarr, duration):
    "convert instrument part to music21 chords"
    part = music21.stream.Part()
    part.append(music21.instrument.Piano())
    part_append_duration_notes(partarr, duration, part) # notes already have duration calculated

    return part


def part_append_duration_notes(partarr, duration, stream):
    "convert instrument part to music21 chords"
    for tidx,t in enumerate(partarr):
        note_idxs = np.where(t > 0)[0] # filter out any negative values (continuous mode)
        if len(note_idxs) == 0: continue
        notes = []
        for nidx in note_idxs:
            note = music21.note.Note(nidx)
            note.duration = music21.duration.Duration(partarr[tidx,nidx]*duration.quarterLength)
            notes.append(note)
        for g in group_notes_by_duration(notes):
            if len(g) == 1:
                stream.insert(tidx*duration.quarterLength, g[0])
            else:
                chord = music21.chord.Chord(g)
                stream.insert(tidx*duration.quarterLength, chord)
    return stream

from itertools import groupby
#  combining notes with different durations into a single chord may overwrite conflicting durations. Example: aylictal/still-waters-run-deep
def group_notes_by_duration(notes):
    "separate notes into chord groups"
    keyfunc = lambda n: n.duration.quarterLength
    notes = sorted(notes, key=keyfunc)
    return [list(g) for k,g in groupby(notes, keyfunc)]
