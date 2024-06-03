# Music Synthesis: Audio Sequence Generation

## Author
Sravan Jayanthi

## Music Synthesis Machine Learning Model
The goal is to create a machine learning model that can generate near-authentic classical music. This repo ablates across several RNN & Attention based approaches to determine their performance a t this task. This model preprocesses the wave based input in the form of midi files and encodes it based on the duration and chord of the notes being played. Then, the music derived data is organized into batches to be used to train the model so that given a sequence of priming notes, it can generate a realistic sequence of chords that form into a melodic song. The newly created music is then decoded so that the encoding utilized in the model are translated back into their representative oscillating musical notes.


### Description
This project contains a model training script and a music prediction script along with assosciated encoding, decoding, and vocabulary implementations.

| File/Directory| Purpose       |
|:-------------:| ------------- | 
| `model.py` | Train the RNN model |
| `predict.py` | Generate sequence of music |
| `vocab.py` | Parameters for encoding |
| `encode.py` | Codify the musical notes |
| `decode.py` | Translate back to notes |
| `transformer.py` | Transformer model implementation |
| `music/` | Classical music input |
| `generated/` | Synthesized music output |
| `training_checkpoints/` | Trained model weights |
| `stats/` | Sample translations |

## Usage
In order to utilize the machine learning model, a repository of music should be identified from which the model will gather its training data from.
1. Port the collection of music in the form of `.mid` or `.midi` files into the `generated/` folder
2. Select a sample to be used as a primer for the model to generate music and place in the `generated/sample` folder
3. Train the GRU/LSTM/Transformer model in `model.py` with the desired training parameters specifying the size and scope of the algorithm
4. Execute the script `model.py` with the requisite dependencies installed, this will generate the model weights which will be stored in the `training_checkpoints/` folder. Command: 
`python model.py -d <file_directory> -m <model:["gru", "lstm", "transformer"]>`
5. Tune the prediction iteration of the trained model in `predict.py` with the desired parameters
6. Execute the script `predict.py` which will sample the primer and synthesize a new song which will be written in the `generated/` folder
7. Play your wonderful artistic piece and enjoy!


### Dependencies
Use the given conda environment export in `environment.yml`
& follow tensorflow [instructions](https://www.tensorflow.org/guide/gpu) for enabling GPU usage on your machine.

## Dataset
Notes about the dataset which can be found [here](https://magenta.tensorflow.org/datasets/maestro):
- 1276 files, 300gb of compressed audio data
- 1.8 mil tokens 

Architecture Parameters
- 312 Vocab size w/ Note toks, Duration toks, Tempo toks, Special toks
- 100 token sequence length 
- 256-dim embedding size

Training Parameters
- 256 Batch Size
- 40 Epochs
- 0.0001 Learning rate
- [0.8, 0.1, 0.1]: Train, Validation, Test split

### Code
Code for predicting notes based off of previously played music using generative random sampling.
```python
    predictions = tf.squeeze(predictions, 0)
    predictions = predictions / TEMPERATURE
    predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
    input_eval = tf.expand_dims([predicted_id], 0)
    music_generated.append(predicted_id)
```

## About MIDI
MIDI (Musical Instrument Digital Interface) representation is a standard protocol that enables electronic musical instruments, computers, and other equipment to communicate, control, and synchronize with each other. MIDI itself does not contain any sound, but rather it is a digital protocol that represents music performance data.

16 Channels for different instruments
-> When a note is played, how long it is held, how hard it is played, and when it is released.


## License
MIT
