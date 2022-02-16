# authors_name = 'Preetham Ganesh'
# project_title = 'Captioning of Images using Attention Mechanism'
# email = 'preetham.ganesh2015@gmail.com'


import os

import numpy as np
import logging
import tensorflow as tf
import unicodedata
import re
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import load_json_file
from utils import save_pickle_file


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


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
    """Pre-processes a sentence.

    Pre-processes the given sentence to remove unwanted characters, lowercase the sentence, etc., and returns the
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


def preprocess_dataset(annotations: dict,
                       data_split: str) -> pd.DataFrame:
    """Pre-processes the dataset.

    Pre-processes the annotations for the current split by processing the image to extract features using the
    pre-trained InceptionV3 model, processing the caption to remove unwanted characters, lowercase the letter, etc.
    Saves the extracted features for each image and returns a dataframe that contains processed captions for the data
    split.

    Args:
        annotations: Dictionary which contains the details for each image, such as image_id and caption.
        data_split: Split the annotations belong to i.e., train or val.

    Returns:
        A Pandas dataframe that contains the updated annotations.
    """
    # Loads the InceptionV3 model pre-trained on Imagenet dataset.
    model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    # Changes the output from the last layer to the last before layer.
    new_model = tf.keras.Model(model.input, model.layers[-1].output)
    new_model.trainable = False
    no_of_images_processed = 0
    annotations_df = pd.DataFrame(annotations['annotations'])
    processed_annotations = {'image_ids': [], 'captions': []}
    # Iterates across the indexes in annotations_df
    for i in range(len(annotations_df)):
        current_image_id = annotations_df['image_id'].iloc[i]
        if current_image_id in processed_annotations['image_ids']:
            continue
        new_image_id = '{}{}'.format(''.join(['0' for _ in range(6 - len(str(current_image_id)))]),
                                     str(current_image_id))
        image_path = '../data/original_data/{}2017/000000{}.jpg'.format(data_split, new_image_id)
        # Pre-process the caption for the current image.
        current_processed_caption = preprocess_sentence(annotations_df['caption'].iloc[i])
        if current_processed_caption == '':
            continue
        # Extracts features from the current image.
        extracted_features = preprocess_image(image_path, new_model)
        processed_annotations['image_ids'].append(current_image_id)
        processed_annotations['captions'].append(current_processed_caption)
        save_pickle_file(extracted_features, '../data/processed_data/images', current_image_id)
        no_of_images_processed += 1
        if no_of_images_processed % 10 == 0:
            print('No. of images processed: {}'.format(no_of_images_processed))
    processed_annotations_df = pd.DataFrame(processed_annotations, columns=['image_ids', 'captions'])
    return processed_annotations_df


def dataset_split_save(train_dataset: pd.DataFrame,
                       validation_dataset: pd.DataFrame) -> None:
    """Splits the original validation dataset and exports the dataset to CSV files.

    Args:
        train_dataset: A pandas dataframe which contains updated annotations for the train dataset.
        validation_dataset: A pandas dataframe which contains updated annotations for the validation dataset.

    Returns:
        None.
    """
    train_dataset.to_csv('../data/processed_data/annotations/train.csv', index=False)
    print('No. of rows in the new train dataset: {}'.format(len(train_dataset)))
    validation_dataset, test_dataset = train_test_split(validation_dataset, test_size=0.5)
    print('No. of rows in the new validation dataset: {}'.format(len(validation_dataset)))
    print('No. of rows in the new test dataset: {}'.format(len(test_dataset)))
    validation_dataset.to_csv('../data/processed_data/annotations/validation.csv', index=False)
    test_dataset.to_csv('../data/processed_data/annotations/test.csv', index=False)


def main():
    print()
    original_train_annotations = load_json_file('../data/original_data/annotations', 'captions_train2017.json')
    print('Loaded the annotations for the train dataset.')
    print()
    new_train_annotations = preprocess_dataset(original_train_annotations, 'train')
    print()
    print('Finished processing images in train dataset')
    print()
    original_validation_annotations = load_json_file('../data/original_data/annotations', 'captions_val2017.json')
    print('Loaded the annotations for the validation dataset')
    print()
    new_validation_annotations = preprocess_dataset(original_validation_annotations, 'val')
    print()
    print('Finished processing images in the validation dataset')
    print()
    dataset_split_save(new_train_annotations, new_validation_annotations)
    print()


if __name__ == '__main__':
    main()
