# authors_name = 'Preetham Ganesh'
# project_title = 'Captioning of Images using Luong Attention Mechanism'
# email = 'preetham.ganesh2015@gmail.com'


import tensorflow as tf
import pickle
import json
import numpy as np
import os


def load_json_file(directory_path: str,
                   file_name: str) -> dict:
    """Loads a JSON file into memory based on the file_name:

    Args:
        file_name: Current name of the dataset split used to load the corresponding JSON file.
        directory_path: Path where the file needs to be saved.

    Returns:
        Loaded JSON file.
    """
    file_path = '{}/{}'.format(directory_path, file_name)
    with open(file_path, 'r') as f:
        captions = json.load(f)
    return captions


def save_pickle_file(file: np.ndarray or dict,
                     directory_path: str,
                     file_name: str) -> None:
    """Saves NumPy array or Dictionary into pickle file for future use.

    Args:
        file: NumPy array or dictionary which needs to be saved.
        directory_path: Path where the file needs to be saved.
        file_name: Name by which the given file should be saved.

    Returns:
        None.
    """
    file_path = '{}/{}.pkl'.format(directory_path, file_name)
    with open(file_path, 'wb') as f:
        pickle.dump(file, f, protocol=2)
    f.close()


def save_json_file(file: dict,
                   directory_path: str,
                   file_name: str) -> None:
    """Saves dictionary into a JSON file for future use.

    Args:
        file: Dictionary which needs to be saved.
        directory_path: Path where the file needs to be saved.
        file_name: Name by which the given file should be saved.

    Returns:
        None.
    """
    file_path = '{}/{}.json'.format(directory_path, file_name)
    with open(file_path, 'w') as f:
        json.dump(file, f, indent=4)


def load_pickle_file(directory_path: str,
                     file_name: str) -> dict:
    """Loads a pickle file into memory based on the file_name:

    Args:
        file_name: Current name of the dataset split used to load the corresponding pickle file.
        directory_path: Path where the file needs to be saved.

    Returns:
        Loaded pickle file.
    """
    file_path = '{}/{}.pkl'.format(directory_path, file_name)
    with open(file_path, 'rb') as f:
        dictionary = pickle.load(f)
    f.close()
    return dictionary


def convert_dataset(dataset: dict) -> tuple:
    """Filters captions with length less than or equal to 40. Converts current datasets into 2 tensors for image ids and
    captions. Pads tensors for captions to ensure uniform size.

    Args:
        dataset: Image ids and captions for the current data split.

    Returns:
        A tuple which contains tensors for image ids and captions.
    """
    image_ids, captions = list(), list()
    for i in range(len(dataset['captions'])):
        # Checks if the length tokenized captions is less than or equal to 40.
        if len(dataset['captions'][i]) <= 40:
            image_ids.append(dataset['image_ids'][i])
            captions.append(dataset['captions'][i])
    # Converts modified list of image ids into a tensor.
    image_ids = tf.convert_to_tensor(image_ids)
    # Converts modified list of captions into a tensor, and pads 0 to the end of each item in the tensor if the length
    # is less than 40.
    captions = tf.keras.preprocessing.sequence.pad_sequences(captions, padding='post')
    return image_ids, captions


def check_directory_existence(directory_path: str,
                              sub_directory: str) -> str:
    """Concatenates directory path and sub_directory. Checks if the directory path exists; if not, it will create the
    directory.

    Args:
        directory_path: Current directory path
        sub_directory: Directory that needs to be checked if it exists or not.

    Returns:
        Newly concatenated directory path
    """
    directory_path = '{}/{}'.format(directory_path, sub_directory)
    # If the concatenated directory path does not exist then the sub directory is created.
    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)
    return directory_path


def shuffle_slice_dataset(image_ids: tf.Tensor,
                          captions: tf.Tensor,
                          batch_size: int) -> tf.data.Dataset:
    """Combines the tensors for the image ids and captions, shuffles them and slices them based on batch size.

    Args:
        image_ids: Tensor which contains image ids for the current data split
        captions: Tensor which contains captions for the current data split.
        batch_size: Batch size for training the current model
    """
    dataset = tf.data.Dataset.from_tensor_slices((image_ids, captions)).shuffle(len(image_ids))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset
