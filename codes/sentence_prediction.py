# authors_name = 'Preetham Ganesh'
# project_title = 'Captioning of Images using Attention Mechanism'
# email = 'preetham.ganesh2015@gmail.com'


import os
import sys

import tensorflow as tf
import tensorflow_datasets as tfds
import logging
import pandas as pd

from utils import choose_encoder_decoder
from utils import load_pickle_file
from utils import convert_dataset
from utils import check_directory_existence
from utils import load_json_file


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def predict_caption(image_features: tf.Tensor,
                    parameters: dict,
                    prediction_stage: str) -> str:
    """Predicts caption for the current image's extracted features using the current model configuration.

    Args:
        image_features: The features extracted for the current image using the pre-trained InceptionV3 model.
        parameters: A dictionary which contains current model configuration details.
        prediction_stage: Location from where the function is being called.

    Returns:
        A string which contains the predicted caption for the current image.
    """
    # Chooses the encoder and decoder based on the parameter configuration.
    encoder, decoder = choose_encoder_decoder(parameters)
    predicted_caption_indexes = []
    # Loads trained tokenizer for captions
    captions_tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(
        '{}/results/utils/captions_tokenizer'.format(prediction_stage))
    # Creates checkpoint for the encoder-decoder model and restores the last saved checkpoint.
    model_directory_path = '{}/results/{}/model_{}'.format(prediction_stage, parameters['attention'],
                                                           parameters['model_number'])
    checkpoint_directory = '{}/checkpoints'.format(model_directory_path)
    checkpoint = tf.train.Checkpoint(encoder=encoder, decoder=decoder)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))
    # Initializes the hidden states from the decoder for each batch.
    decoder_hidden_states = decoder.initialize_hidden_states(1, parameters['rnn_size'])
    # First decoder input batch contains just the start token index.
    decoder_input = tf.expand_dims([parameters['start_token_index']], 0)
    encoder_out = encoder(image_features, False)
    # Passes the encoder features into the decoder for all words in the captions.
    for i in range(1, 100):
        prediction, decoder_hidden_states = decoder(decoder_input, decoder_hidden_states, encoder_out, False)
        predicted_id = tf.argmax(prediction[0]).numpy()
        # Uses teacher forcing method to pass next target word as input into the decoder.
        decoder_input = tf.expand_dims([predicted_id], 0)
        predicted_caption_indexes.append(predicted_id)
        if captions_tokenizer.vocab_size + 1 == predicted_id:
            break
    predicted_caption_indexes = tf.convert_to_tensor(predicted_caption_indexes)
    predicted_caption = captions_tokenizer.decode([i for i in predicted_caption_indexes.numpy()[1:-1]])
    return predicted_caption


def predict_caption_dataset(attention: str,
                            model: str,
                            data_split: str) -> None:
    annotations = load_pickle_file('../data/tokenized_data/annotations', data_split)
    image_ids, captions = convert_dataset(annotations)
    directory_path = check_directory_existence('../results', attention)
    directory_path = check_directory_existence(directory_path, 'model_{}'.format(model))
    directory_path = check_directory_existence(directory_path, 'utils')
    parameters = load_json_file(directory_path, 'parameters')
    captions_tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(
        '{}/results/utils/captions_tokenizer'.format('..'))
    for i in range(1):
        print(image_ids[i].numpy())
        current_image_features = load_pickle_file('../data/processed_data/images', str(image_ids[i].numpy()))
        predict_caption(current_image_features, parameters, '..')
        current_caption = captions[i, :]
        target_caption = captions_tokenizer.decode([j for j in current_caption[1:-1] if j != captions_tokenizer.vocab_size + 1])
        print(target_caption)


def main():
    attention = sys.argv[1]
    model = sys.argv[2]
    data_split = sys.argv[3]
    predict_caption_dataset(attention, model, data_split)


if __name__ == '__main__':
    main()
