# authors_name = 'Preetham Ganesh'
# project_title = 'Captioning of Images using Attention Mechanism'
# email = 'preetham.ganesh2015@gmail.com'


import os

import tensorflow as tf
import logging
import numpy as np
import json
from flask import Flask
from flask import request
from flask import render_template
from flask import send_from_directory
import tensorflow_datasets as tfds

from codes.bahdanau_attention_model import Encoder
from codes.bahdanau_attention_model import BahdanauDecoder1
from codes.bahdanau_attention_model import BahdanauDecoder2
from codes.bahdanau_attention_model import BahdanauDecoder3
from codes.luong_attention_model import LuongDecoder1
from codes.luong_attention_model import LuongDecoder2
from codes.luong_attention_model import LuongDecoder3


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

app = Flask(__name__)
app_root_directory = os.path.dirname(os.path.abspath(__file__))


def preprocess_image(image_path: str,
                     model: tf.keras.Model) -> np.ndarray:
    """Pre-processes the given image to extract features.

    Reads the image, pre-processes it, and uses the pre-trained InceptionV3 to extract features from the
    pre-processed image.

    Args:
        image_path: Path to the image that should be read.
        model: The pre-trained InceptionV3 model used to extract features from the image.

    Returns:
        The features extracted from the image using the pre-trained InceptionV3 model.
    """
    # Reads image using the file name
    image = tf.io.read_file(image_path)
    # Decodes the read image into a RGB image and resizes it to the size of (299, 299).
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (299, 299))
    # Pre-processes the resized image based on InceptionV3 input requirements.
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    image = tf.convert_to_tensor([image])
    # Extracts features from the pre-processed image using the pre-trained InceptionV3 model.
    image = model(image)
    image = tf.reshape(image, [image.shape[0], -1, image.shape[3]])
    return image.numpy()


def load_json_file(directory_path: str,
                   file_name: str) -> dict:
    """Loads a JSON file into memory based on the file_name:

    Args:
        file_name: Current name of the dataset split used to load the corresponding JSON file.
        directory_path: Path where the file needs to be saved.

    Returns:
        Loaded JSON file.
    """
    file_path = '{}/{}.json'.format(directory_path, file_name)
    with open(file_path, 'r') as out_file:
        captions = json.load(out_file)
    out_file.close()
    return captions


def choose_encoder_decoder(parameters) -> tuple:
    """Uses attention and model number keys in the parameters, chooses the encoder and decoder model.

    Args:
        parameters: A dictionary which contains current model configuration details.

    Returns:
         A tuple which contains the objects for the encoder and decoder models
    """
    encoder = Encoder(parameters['embedding_size'], parameters['dropout_rate'])
    if parameters['attention'] == 'bahdanau_attention' and parameters['model_number'] == 1:
        decoder = BahdanauDecoder1(parameters['embedding_size'], parameters['rnn_size'],
                                   parameters['target_vocab_size'], parameters['dropout_rate'])
    elif parameters['attention'] == 'bahdanau_attention' and parameters['model_number'] == 2:
        decoder = BahdanauDecoder2(parameters['embedding_size'], parameters['rnn_size'],
                                   parameters['target_vocab_size'], parameters['dropout_rate'])
    elif parameters['attention'] == 'bahdanau_attention' and parameters['model_number'] == 3:
        decoder = BahdanauDecoder3(parameters['embedding_size'], parameters['rnn_size'],
                                   parameters['target_vocab_size'], parameters['dropout_rate'])
    elif parameters['attention'] == 'luong_attention' and parameters['model_number'] == 1:
        decoder = LuongDecoder1(parameters['embedding_size'], parameters['rnn_size'], parameters['dropout_rate'],
                                parameters['target_vocab_size'])
    elif parameters['attention'] == 'luong_attention' and parameters['model_number'] == 2:
        decoder = LuongDecoder2(parameters['embedding_size'], parameters['rnn_size'], parameters['dropout_rate'],
                                parameters['target_vocab_size'])
    elif parameters['attention'] == 'luong_attention' and parameters['model_number'] == 3:
        decoder = LuongDecoder3(parameters['embedding_size'], parameters['rnn_size'], parameters['dropout_rate'],
                                parameters['target_vocab_size'])
    return encoder, decoder


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
    model_directory_path = 'results/{}/model_{}'.format(parameters['attention'], parameters['model_number'])
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


@app.route('/index', methods=['POST'])
def image_upload():
    """Uploads image from directory, performs preprocessing and predicts the label.

        Args:
            None.

        Returns:
            Rendered template for the complete page with image and the predicted label..

    """
    # Creates a directory for storing the uploaded image if it does not exist.
    directory_path = '{}/{}'.format(app_root_directory, 'data/images')
    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)
    uploaded_file = request.files['upload_file']
    attention = 'luong_attention'
    model = 2
    # Saves the uploaded in the server's designated location.
    if uploaded_file.filename != '':
        image_path = 'data/images/{}'.format(uploaded_file.filename)
        uploaded_file.save('data/images/{}'.format(uploaded_file.filename))
        # Loads the InceptionV3 model pre-trained on Imagenet dataset.
        feature_extractor_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
        extracted_features = preprocess_image(image_path, feature_extractor_model)
        parameters = load_json_file('results/{}/model_{}/utils'.format(attention, model), 'parameters')
        captions_tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file('results/utils/captions_tokenizer')
        predicted_caption = predict_caption(extracted_features, parameters, captions_tokenizer)
        return render_template('complete.html', image_name=uploaded_file.filename, caption=predicted_caption)
    else:
        return render_template('error.html')


@app.route('/upload/<filename>')
def send_image(filename):
    """Sends saved image from directory using uploaded image filename.

        Args:
            filename: A string which contains the filename for the uploaded image.

        Returns:
            Saved image from directory.
    """
    return send_from_directory('data/images', filename)


@app.route("/")
def index():
    """Renders template for index page.

        Args:
            None.

        Returns:
            Rendered template for the index page.
    """
    return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True)
