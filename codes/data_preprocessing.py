# authors_name = 'Preetham Ganesh'
# project_title = 'Captioning of Images using Luong Attention'
# email = 'preetham.ganesh2015@gmail.com'
import numpy as np
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


def retrieve_image_names(data_split: str) -> list:
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
                     model: tf.keras.Model) -> np.ndarray:
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
    return image.numpy()


def remove_html_markup(sentence: str) -> str:
    """Removes HTML markup from sentences given as input and returns processed sentences.

        Args:
            sentence: Input sentence from which HTML markups should be removed (if it exists).

        Returns:
            Processed sentence from which HTML markups are removed (if it exists).
    """
    tag = False
    quote = False
    processed_sentence = ''
    for i in range(len(sentence)):
        if sentence[i] == '<' and not quote:
            tag = True
        elif sentence[i] == '>' and not quote:
            tag = False
        elif (sentence[i] == '"' or sentence[i] == "'") and tag:
            quote = not quote
        elif not tag:
            processed_sentence += sentence[i]
    return processed_sentence


def preprocess_sentence(sentence: str) -> str:
    """Pre-processes the given sentence to remove unwanted characters, lowercase the sentence, etc., and returns the
    processed sentence.

        Args:
            sentence: Input sentence which needs to be processed.

        Returns:
            The processed sentence that does not have unwanted characters, lowercase letters, and many more.
    """
    # Removes HTML marksups from the sentence.
    sentence = remove_html_markup(sentence)
    # Lowercases the letters in the sentence, and removes spaces from the beginning and the end of the sentence.
    sentence = sentence.lower().strip()
    # Converts UNICODE characters to ASCII format.
    sentence = ''.join(i for i in unicodedata.normalize('NFD', sentence) if unicodedata.category(i) != 'Mn')
    # Removes characters which does are not in -!$&(),./%0-9:;?a-z¿¡€'
    sentence = re.sub(r"[^-!$&(),./%0-9:;?a-z¿¡€'\"]+", " ", sentence)
    # Converts words or tokens like 1st to 1 st, to simplify the tokens.
    sentence = re.sub(r'(\d)th', r'\1 th', sentence, flags=re.I)
    sentence = re.sub(r'(\d)st', r'\1 st', sentence, flags=re.I)
    sentence = re.sub(r'(\d)rd', r'\1 rd', sentence, flags=re.I)
    sentence = re.sub(r'(\d)nd', r'\1 nd', sentence, flags=re.I)
    # Adds space between punctuations to simplify the tokens.
    punctuations = list("-!$&(),./%:;?¿¡€'")
    for i in range(len(punctuations)):
        sentence = sentence.replace(punctuations[i], ' {} '.format(punctuations[i]))
    # Removes spaces from the beginning and the end of the sentence
    sentence = sentence.strip()
    # Removes unwanted spaces from the sentence.
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence


def main():
    original_validation_captions = load_json_file('../data/original_data/annotations', 'captions_val2017.json')
    #print(original_validation_captions)
    original_validation_image_names = retrieve_image_names('val')
    print(preprocess_sentence('Preetham was the 1st student in     class <html> slkdfjslkj </html>'))


if __name__ == '__main__':
    main()
