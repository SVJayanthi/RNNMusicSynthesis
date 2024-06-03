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

from model import build_gru_model, build_lstm_model, build_transformer_model, SEQ_LENGTH

import argparse
from datetime import datetime
import sys
from pathlib import Path

# The embedding dimension
EMBEDDING_DIM = 256
# Number of RNN units
RNN_UNITS = 1024
# Number of notes to generate
NUM_GENERATE = 1500
# Predictability of generated output
TEMPERATURE = 1.0
# Random seed
RANDOM_SEED = 3

    
def build_arg_parser():    
    parser = argparse.ArgumentParser(description="Inference with Music Synthesis Model")
    parser.add_argument("-d", "--dir", type=str, help="Directory of model checkpoint", required=True)
    parser.add_argument("-s", "--samplefile", type=str, help="Directory to midi sample file", required=True)
    return parser.parse_args()


if __name__ == '__main__':
    tf.get_logger().setLevel('ERROR')
    tf.random.set_seed(RANDOM_SEED)
    args = build_arg_parser()
    print(args)
    
    checkpoint_dir = args.dir
    musical_sample_file = args.samplefile
    
    # Get tokenizer and tokenize musical sample
    vocab = MusicVocab.create()
    idxenc_sample = file2idxenc(musical_sample_file, vocab)
    idxenc_sample = np.array(idxenc_sample)
    
    # Select from the first third of the sample the last seq. length # of tokens
    sample_index = int(len(idxenc_sample) / 3)
    input_eval_idx = idxenc_sample[sample_index-SEQ_LENGTH:sample_index]
    input_eval = tf.expand_dims(input_eval_idx, 0)
    
    if "transformer" in checkpoint_dir:
        model = build_transformer_model(vocab_size = len(vocab.itos))
            
        # Empty list to store our results
        music_generated = []
        for i in range(NUM_GENERATE):
            predictions = model(input_eval, training=False)
            # Select the last token from the `seq_len` dimension.
            predictions = predictions[:, -1:, :]
            predictions = tf.squeeze(predictions, 0)
        
            # using a categorical distribution to predict the character returned by the model
            predictions = predictions / TEMPERATURE
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

            # Concatenate the `predicted_id` to the output which is given to decoder as its input.
            input_eval = tf.concat([input_eval[:, :-1], [[predicted_id]]], axis=1)
            music_generated.append(predicted_id)

            if predicted_id == vocab.eos_idx:
                break
        
    else:
        if "gru" in checkpoint_dir:
            model = build_gru_model(vocab_size = len(vocab.itos), batch_size=1)
        elif "lstm" in checkpoint_dir:
            model = build_lstm_model(vocab_size = len(vocab.itos), batch_size=1)
        else:
            raise Exception('Checkpoint not among supported models: "gru", "lstm", "transformer"')

        model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
        print(model.summary()) 
      
        # Empty string to store our results
        music_generated = []
                
        # Here batch size == 1
        model.reset_states()
        for i in range(NUM_GENERATE):
            predictions = model(input_eval)
            predictions = tf.squeeze(predictions, 0)
        
            # using a categorical distribution to predict the character returned by the model
            predictions = predictions / TEMPERATURE
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        
            # We pass the predicted character as the next input to the model
            # along with the previous hidden state
            input_eval = tf.expand_dims([predicted_id], 0)
            music_generated.append(predicted_id)
            
            if predicted_id == vocab.eos_idx:
                break
    
    predict_sample = input_eval_idx.tolist()
    predict_sample.extend(music_generated)
    generated_stream = idxenc2stream(predict_sample, vocab)
    
    # Get current datetime
    now = datetime.now()
    datetime_str = now.strftime("%y_%m_%d_%H_%M_%S")
    musical_sample_file_stem_name = Path(musical_sample_file).stem
    
    sample_name = datetime_str + "_" + musical_sample_file_stem_name + ".mid"
    save_directory = os.path.join(checkpoint_dir, "generated")
    os.makedirs(save_directory, exist_ok=True)
    generated_stream.write('midi', fp= os.path.join(save_directory, sample_name))