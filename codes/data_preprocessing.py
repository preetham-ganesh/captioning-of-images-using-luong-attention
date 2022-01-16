# authors_name = 'Preetham Ganesh'
# project_title = 'Captioning of Images using Luong Attention'
# email = 'preetham.ganesh2015@gmail.com'


import tensorflow as tf
import os
import unicodedata
import re
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import load_json_file


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def retrieve_image_names(data_split: str):
    """Retrieves image names in the current data split directory.

        Args:
            data_split: String which contains the data split name i.e. train or val.

        Returns:
            List of image names with their corresponding path.
    """
    directory_path = '../data/original_data'
    image_names = []
    # Iterates across image names in the current data split directory.
    for i in os.listdir('{}/{}2017'.format(directory_path, data_split)):
        # Checks if the current image name is classified as file. If yes, then appended to the final list of images.
        if os.path.isfile('{}/{}2017/{}'.format(directory_path, data_split, i)):
            image_names.append('{}/{}2017/{}'.format(directory_path, data_split, i))
    return image_names


def preprocess_image(file_name: str,
                     model: tf.keras.Model):
    """Reads the image, pre-processes it, and uses the pre-trained InceptionV3 to extract features from the
    pre-processed image.

        Args:
            file_name: Name of the image used to read the image.
            model: The pre-trained InceptionV3 model used to extract features from the image.

        Returns:
            The features extracted from the image using the pre-trained InceptionV3 model.
    """
    # Reads image using the file name
    image = tf.io.read_file(file_name)
    # Decodes the read image into a RGB image and resizes it to the size of (299, 299).
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (299, 299))
    # Pre-processes the resized image based on InceptionV3 input requirements.
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    image = tf.convert_to_sensor([image])
    # Extracts features from the pre-processed image using the pre-trained InceptionV3 model.
    image = model(image)
    image = tf.reshape(image, [image.shape[0], -1, image.shape[3]])
    return image


def main():
    original_validation_captions = load_json_file('../data/original_data/annotations', 'captions_val2017.json')
    original_validation_image_names = retrieve_image_names('val')
    print(original_validation_image_names)


if __name__ == '__main__':
    main()
