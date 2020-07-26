# RNNMusicSynthesis

## Author
Sravan Jayanthi

## Music Synthesis Machine Learning Model
The goal is to create a machine learning model that can generate near-authentic classical music. This model is designed utilizing Recurrent Neural Networks as the first iteration in a series of different models to be tested to accomplish the goal. This model preprocesses the wave based input in the form of midi files and encodes it based on the duration and chord of the notes being played. Then, the music derived data is organized into batches to be used to train the Recurrent Nerual Network model so that given a sequence of priming notes, it can generate a realistic sequence of chords that form into a melodic song. The newly created music is then decoded so that the encoding utilized in the model are translated back into their representative oscillating musical notes.


### Description
This project contains a model training script and a music prediction script along with assosciated encoding, decoding, and vocabulary implementations.

| File/Directory| Purpose       |
|:-------------:| ------------- | 
| `model.py` | Train the RNN model |
| `predict.py` | Generate sequence of music |
| `vocab.py` | Parameters for encoding |
| `encode.py` | Codify the musical notes |
| `decode.py` | Translate back to notes |
| `music/` | Classical music input |
| `generated/` | Synthesized music output |
| `training_checkpoints/` | Trained model weights |
| `stats/` | Sample translations |

## Usage
In order to utilize the machine learning model, a repository of music should be identified from which the model will gather its training data from.
1. Port the collection of music in the form of `.mid` or `.midi` files into the `generated/` folder
2. Select a sample to be used as a primer for the model to generate music and place in the `generated/sample` folder
3. Tune the RNN model in `model.py` with the desired training parameters specifying the size and scope of the algorithm
4. Execute the script `model.py` with the requisite dependencies installed, this will generate the model weights which will be stored in the `training_checkpoints/` folder
5. Tune the prediction iteration of the trained model in `predict.py` with the desired parameters
6. Execute the script `predict.py` which will sample the primer and synthesize a new song which will be written in the `generated/` folder
7. Play your wonderful artistic piece and enjoy!


### Dependencies
- Tensorflow
- Music21


### Code
Sample code of predicting notes based off of previously played music.
```python
    predictions = tf.squeeze(predictions, 0)
    predictions = predictions / TEMPERATURE
    predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
    input_eval = tf.expand_dims([predicted_id], 0)
    music_generated.append(predicted_id)
```

## License
MIT
