# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 22:57:42 2020

@author: srava
"""

import tensorflow as tf

import os
import numpy as np

from vocab import MusicVocab
from encode import file2idxenc


# Max Sequence Length
SEQ_LENGTH = 100
# Batch size
BATCH_SIZE = 64
# Buffer size to shuffle the dataset
BUFFER_SIZE = 10000
# The embedding dimension
EMBEDDING_DIM = 256
# Number of RNN units
RNN_UNITS = 1024
# Iterations to Train
EPOCHS=10


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
    vocab = MusicVocab.create()
    
    
    train_files = []
    train_dir = "music/"   
    for file_name in os.listdir(train_dir):
        if file_name.endswith('.mid') or file_name.endswith('.midi'):
            train_files.append(file_name)
    
        
    idxenc_data = []
    for music_file in train_files:
        idxenc_data.extend(file2idxenc(train_dir + music_file, vocab))
    idxenc_data = np.array(idxenc_data)
    
    
    examples_per_epoch = len(idxenc_data)//(SEQ_LENGTH+1)
    
    print("Input length: ", idxenc_data.size)
    
    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'
    
    # Create training examples / targets
    char_dataset = tf.data.Dataset.from_tensor_slices(idxenc_data)
    sequences = char_dataset.batch(SEQ_LENGTH+1, drop_remainder=True)
    
    dataset = sequences.map(lambda x: (x[:-1], x[1:]))
    
    dataset = dataset.shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    
    model = build_model(vocab_size = len(vocab.itos), embedding_dim=EMBEDDING_DIM,
                        rnn_units=RNN_UNITS, batch_size=BATCH_SIZE)    
    
    print(model.summary())
    
    model.compile(optimizer='adam', loss=loss)
    
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    
    checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)
    
    
    history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])