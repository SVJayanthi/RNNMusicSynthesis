# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 22:57:42 2020

@author: srava
"""

import tensorflow as tf
import os
import numpy as np
import argparse
import json

from datetime import datetime
from vocab import MusicVocab
from transformer import Transformer
from loader import load_files_parallel, load_files

# Max Sequence Length
SEQ_LENGTH = 100
# Batch size
BATCH_SIZE = 256
# Buffer size to shuffle the dataset
BUFFER_SIZE = 10000
# The embedding dimension
EMBEDDING_DIM = 256
# Number of RNN units
RNN_UNITS = 1024
# Iterations to Train
EPOCHS = 40
# Learning rate
LEARNING_RATE = 0.0001
# Random seed to split data
RANDOM_SEED = 3


def build_gru_model(vocab_size, embedding_dim=EMBEDDING_DIM, rnn_units=RNN_UNITS, batch_size=BATCH_SIZE):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform',
                            kernel_initializer='glorot_uniform'),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

def build_lstm_model(vocab_size, embedding_dim=EMBEDDING_DIM, rnn_units=RNN_UNITS, batch_size=BATCH_SIZE):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.LSTM(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform',
                            kernel_initializer='glorot_uniform'),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

def build_transformer_model(vocab_size, num_layers=4, d_model=128, dff=512, num_heads=12, dropout_rate=0.05):
    model = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=vocab_size,
        target_vocab_size=vocab_size,
        dropout_rate=dropout_rate)
    return model


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

def masked_loss(label, pred):
    mask = label != 0
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
    loss = loss_object(label, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
    return loss


def split_dataset(dataset, val_split=0.1, test_split=0.1):
    test_split_idx = 1 - test_split
    val_split_idx = test_split_idx - val_split
    # Split the train, val, & test
    total_batches = dataset.cardinality().numpy()
    train_split_index = int(val_split_idx * total_batches)
    val_split_index = int(test_split_idx * total_batches)

    # Split the dataset into train, validation, and test datasets
    train_dataset = dataset.take(train_split_index)
    val_dataset = dataset.skip(train_split_index).take(val_split_index - train_split_index)
    test_dataset = dataset.skip(val_split_index)
    return train_dataset, val_dataset, test_dataset
    

def build_arg_parser():    
    parser = argparse.ArgumentParser(description="Train Music Synthesis Model")
    parser.add_argument("-d", "--dir", type=str, help="Directory of files", required=True)
    parser.add_argument("-m", "--model", default=["gru"], type=str, nargs="+", help="Type of models options: gru, lstm, transformer", required=False)
    return parser.parse_args()


if __name__ == '__main__':
    tf.random.set_seed(RANDOM_SEED)
    args = build_arg_parser()
    print(args)
    
    models = args.model
    train_dir = args.dir
    
    vocab = MusicVocab.create()
    
    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'
    
    # Load dataset
    dataset = load_files_parallel(
        train_dir, 
        vocab, 
        seq_length=SEQ_LENGTH, 
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE
    )
    
    # # Get dataset splits
    train_dataset, val_dataset, test_dataset = split_dataset(dataset)
    
    for model_name in models:
        if model_name == "transformer":
            model = build_transformer_model(vocab_size = len(vocab.itos))
            optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
            model.compile(
                optimizer=optimizer,
                loss=masked_loss,
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='sparse_categorical_accuracy')])
            
        else:
            if model_name == "gru":
                model = build_gru_model(vocab_size = len(vocab.itos), embedding_dim=EMBEDDING_DIM,
                                    rnn_units=RNN_UNITS, batch_size=BATCH_SIZE)
            elif model_name == "lstm":
                model = build_lstm_model(vocab_size = len(vocab.itos), embedding_dim=EMBEDDING_DIM,
                                rnn_units=RNN_UNITS, batch_size=BATCH_SIZE)
            else:
                raise Exception("Not implemented yet")
            
            optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
            model.compile(optimizer=optimizer, 
                        loss=loss,
                        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='sparse_categorical_accuracy')])
            
            print(model.summary())
        
        # Name of the checkpoint files 
        now = datetime.now()
        datetime_str = now.strftime("%y_%m_%d_%H_%M_%S")
        save_directory = os.path.join(checkpoint_dir, model_name, datetime_str)
        checkpoint_prefix = os.path.join(save_directory, "ckpt_{epoch}")
        
        checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix,
            save_weights_only=True)
        class TerminateOnNaN(tf.keras.callbacks.Callback):
            def on_batch_end(self, batch, logs={}):
                loss = logs.get('loss')
                if loss is not None and np.isnan(loss):
                    print(loss)
                    print('Batch %d: Invalid loss, terminating training' % (batch))
                    self.model.stop_training = True
        
        history = model.fit(
            train_dataset, 
            epochs=EPOCHS,
            validation_data=val_dataset,
            validation_steps=None,
            callbacks=[checkpoint_callback, TerminateOnNaN()]
        )
        
        test_loss, test_acc = model.evaluate(test_dataset)
        print(f'Test loss: {test_loss}, Test accuracy: {test_acc}')
        history.history["test_loss"] = test_loss
        history.history["test_accuracy"] = test_acc

        history_path = os.path.join(save_directory, "history.json")
        with open(history_path, 'w') as f:
            json.dump(history.history, f)

        