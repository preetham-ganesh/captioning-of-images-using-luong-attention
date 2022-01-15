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
    directory_path = '../data/original_data/annotations/'
    with open('{}/{}'.format(directory_path, file_name), 'r') as f:
        captions = json.load(f)
    return captions


def main():
    x = 0


if __name__ == '__main__':
    main()
