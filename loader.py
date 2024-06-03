from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm
from encode import file2idxenc
import numpy as np
import os
import tensorflow as tf

def find_midi_files(directory):
    midi_files = []
    for root, _, files in os.walk(directory):
        for file_name in files:
            if file_name.endswith('.mid') or file_name.endswith('.midi'):
                full_path = os.path.join(root, file_name)
                midi_files.append(full_path)
    return midi_files

def _helper_process(music_file, vocab):
    return file2idxenc(music_file, vocab)

def process_file(train_files, vocab, max_workers=10):
    idxenc_data = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_helper_process, file, vocab): file for file in train_files}
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            idxenc_data.extend(future.result())
    return np.array(idxenc_data)

def build_dataset(idxenc_data, seq_length=100, buffer_size=10000, batch_size=256):
    # Create training examples / targets
    char_dataset = tf.data.Dataset.from_tensor_slices(idxenc_data)
    sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
    
    dataset = sequences.map(lambda x: (x[:-1], x[1:]))
    
    dataset = dataset.shuffle(buffer_size)
    return dataset.batch(batch_size, drop_remainder=True)

def clean_encodings(idxenc_data, vocab):
    # Validate the tokenization
    assert not np.isnan(idxenc_data).any()
    # Remove invalid tokens
    min_id, max_id = 0, len(vocab.itos)  
    idxenc_data = idxenc_data[(idxenc_data >= min_id) & (idxenc_data < max_id)]
    return idxenc_data

def load_files_parallel(directory, vocab,
                        seq_length=100, 
                        buffer_size=10000, 
                        batch_size=256,
                        logging=True,
                        max_workers=8):
    train_files = find_midi_files(directory)
    
    if logging:
        print("Files count: ", len(train_files))
    
    idxenc_data = process_file(train_files, vocab, max_workers=max_workers)
    if logging:
        print("Input length: ", idxenc_data.size)
    idxenc_data = clean_encodings(idxenc_data, vocab)
    return build_dataset(idxenc_data, seq_length, buffer_size, batch_size)
    
        
        
def load_files(directory, vocab, 
                        seq_length=100, 
                        buffer_size=10000, 
                        batch_size=256,
                        logging=True):
    train_files = find_midi_files(directory)
    
    if logging:
        print("Files count: ", len(train_files))
    
    idxenc_data = []
    for music_file in train_files:
        idxenc_data.extend(file2idxenc(music_file, vocab))
    idxenc_data = np.array(idxenc_data)
    idxenc_data = clean_encodings(idxenc_data, vocab)
    
    return build_dataset(idxenc_data, seq_length, buffer_size, batch_size)