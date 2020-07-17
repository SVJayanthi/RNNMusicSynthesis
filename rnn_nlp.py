# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 19:43:43 2020

@author: srava
"""

import tensorflow as tf

import os

from vocab import MusicVocab
from decode import idxenc2stream
from encode import file2idxenc

# Train model
TRAIN = False
# Max Sequence Length
SEQ_LENGTH = 100
# Batch size
BATCH_SIZE = 17
# Buffer size to shuffle the dataset
BUFFER_SIZE = 10000
# The embedding dimension
EMBEDDING_DIM = 256
# Number of RNN units
RNN_UNITS = 1024
# Iterations to Train
EPOCHS=10
# Number of notes to generate
NUM_GENERATE = 1000
# Predictability of generated output
TEMPERATURE = 1.0

def target_batch(dataset, model):
    example_batch_predictions = None
    target_batch = None
    for input_example_batch, target_example_batch in dataset.take(1):
        example_batch_predictions = model(input_example_batch)
        target_batch = target_example_batch
        print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
    
    sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
    sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
    
    example_batch_loss  = loss(target_batch, example_batch_predictions)
    print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
    print("scalar_loss:      ", example_batch_loss.numpy().mean())

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
    midi_file = "music/bwv772.mid"
    vocab = MusicVocab.create()
    idxenc = file2idxenc(midi_file, vocab)
    
    examples_per_epoch = len(idxenc)//(SEQ_LENGTH+1)
    
    print("Input length: ", idxenc.size)
    
    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'
    
    if (TRAIN):
        # Create training examples / targets
        char_dataset = tf.data.Dataset.from_tensor_slices(idxenc)
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
    
    model = build_model(len(vocab.itos), EMBEDDING_DIM, RNN_UNITS, batch_size=1)

    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    
    model.build(tf.TensorShape([1, None]))
    
    print(model.summary()) 
    
    # Converting our start string to numbers (vectorizing)
    input_eval = idxenc[:10]
    input_eval = tf.expand_dims(input_eval, 0)
      
    # Empty string to store our results
    music_generated = []
            
    # Here batch size == 1
    model.reset_states()
    for i in range(NUM_GENERATE):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)
      
        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / TEMPERATURE
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
      
        # We pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)
      
        music_generated.append(predicted_id)
    
    generated_stream = idxenc2stream(music_generated, vocab)
    generated_stream.write('midi', fp='./generated/synthesized_song.mid')
