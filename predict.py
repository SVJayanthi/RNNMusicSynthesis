# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 19:43:43 2020

@author: srava
"""

import tensorflow as tf

import os
import numpy as np

from vocab import MusicVocab
from decode import idxenc2stream
from encode import file2idxenc

from model import build_model, loss


# The embedding dimension
EMBEDDING_DIM = 256
# Number of RNN units
RNN_UNITS = 1024
# Number of notes to generate
NUM_GENERATE = 1500
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
    

if __name__ == '__main__':
    vocab = MusicVocab.create()
    
    sample_files = []
    sample_dir = "music/sample/"   
    for file_name in os.listdir(sample_dir):
        if file_name.endswith('.mid') or file_name.endswith('.midi'):
            sample_files.append(file_name)
    
            
    idxenc_sample = []
    for music_file in sample_files:
        idxenc_sample.extend(file2idxenc(sample_dir + music_file, vocab))
    idxenc_sample = np.array(idxenc_sample)
        
    
    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'
    
    model = build_model(len(vocab.itos), EMBEDDING_DIM, RNN_UNITS, batch_size=1)

    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    
    model.build(tf.TensorShape([1, None]))
    
    print(model.summary()) 
    
    # Converting our start string to numbers (vectorizing)
    sample_index = int(len(idxenc_sample) / 3)
    input_eval_idx = idxenc_sample[:sample_index]
    input_eval = tf.expand_dims(input_eval_idx, 0)
      
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
    
    predict_sample = input_eval_idx.tolist()
    predict_sample.extend(music_generated)
    generated_stream = idxenc2stream(predict_sample, vocab)
    generated_stream.write('midi', fp=('./generated/synthesized_' + sample_files[0].replace('.mid', '') + '.mid'))