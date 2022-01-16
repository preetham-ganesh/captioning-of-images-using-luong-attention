# authors_name = 'Preetham Ganesh'
# project_title = 'Captioning of Images using Luong Attention'
# email = 'preetham.ganesh2015@gmail.com'


import tensorflow as tf
import json
import os
import unicodedata
import re
import logging
import pickle
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def load_json_file(file_name: str):
    """Loads a JSON file into memory based on the file_name:

        Args:
            file_name: Current name of the dataset split used to load the corresponding JSON file.

        Returns:
            Loaded JSON file which contains the image file names and captions in the current split of the dataset.
    """
    directory_path = '../data/original_data/annotations'
    with open('{}/{}'.format(directory_path, file_name), 'r') as f:
        captions = json.load(f)
    return captions


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


def main():
    original_validation_captions = load_json_file('captions_val2017.json')
    original_validation_image_names = retrieve_image_names('val')
    print(original_validation_image_names)


if __name__ == '__main__':
    main()
