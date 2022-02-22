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
                    captions_tokenizer: tfds.deprecated.text.SubwordTextEncoder) -> str:
    """Predicts caption for the current image's extracted features using the current model configuration.

    Args:
        image_features: The features extracted for the current image using the pre-trained InceptionV3 model.
        parameters: A dictionary which contains current model configuration details.
        captions_tokenizer: A TFDS tokenizer trained on the captions in the trained dataset.

    Returns:
        A string which contains the predicted caption for the current image.
    """
    # Chooses the encoder and decoder based on the parameter configuration.
    encoder, decoder = choose_encoder_decoder(parameters)
    predicted_caption_indexes = []
    # Creates checkpoint for the encoder-decoder model and restores the last saved checkpoint.
    model_directory_path = '../results/{}/model_{}'.format(parameters['attention'], parameters['model_number'])
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
    # Decodes the prediction captions by getting sub-tokens from the trained captions tokenizer
    predicted_caption = captions_tokenizer.decode([i for i in predicted_caption_indexes.numpy()[1:-1]])
    return predicted_caption


def predict_caption_dataset(attention: str,
                            model: str,
                            data_split: str) -> None:
    """Predicts captions for all the images in the dataset for the current data split.

    Args:
        attention: Name of the current attention.
        model: Name of the current model.
        data_split: Name of the data for the current split.

    Returns:
        None.
    """
    annotations = load_pickle_file('../data/tokenized_data/annotations', data_split)
    image_ids, captions = convert_dataset(annotations)
    parameters = load_json_file('../results/{}/model_{}/utils'.format(attention, model), 'parameters')
    # Loads the trained tokenizer for the captions.
    captions_tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(
        '../results/utils/captions_tokenizer')
    current_data_split_predictions = pd.DataFrame(columns=['image_id', 'target_caption', 'predicted_caption'])
    for i in range(image_ids.shape[0]):
        current_image_features = load_pickle_file('../data/processed_data/images', str(image_ids[i].numpy()))
        current_predicted_caption = predict_caption(current_image_features, parameters, captions_tokenizer)
        current_target_caption_indexes = captions[i, :]
        # Decodes the prediction captions by getting sub-tokens from the trained captions tokenizer
        current_target_caption = captions_tokenizer.decode([j for j in current_target_caption_indexes[1:-1]
                                                            if j != captions_tokenizer.vocab_size + 1])
        print('Image id: {}'.format(str(image_ids[i].numpy())))
        print('Target caption: {}'.format(current_target_caption))
        print('Predicted caption: {}'.format(current_predicted_caption))
        current_predictions = {'image_id': str(image_ids[i].numpy()), 'target_caption': current_target_caption,
                               'predicted_caption': current_predicted_caption}
        current_data_split_predictions = current_data_split_predictions.append(current_predictions, ignore_index=True)
        print()
    directory_path = check_directory_existence('../results/{}/model_{}'.format(attention, model), 'predictions')
    current_data_split_predictions.to_csv('{}/{}.csv'.format(directory_path, data_split), index=False)
    print('Finished predicting captions for {} model_{} for the {} data.'.format(attention, model, data_split))
    print()


def main():
    print()
    attention = sys.argv[1]
    model = sys.argv[2]
    data_split = sys.argv[3]
    predict_caption_dataset(attention, model, data_split)


if __name__ == '__main__':
    main()
