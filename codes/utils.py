# authors_name = 'Preetham Ganesh'
# project_title = 'Captioning of Images using Luong Attention Mechanism'
# email = 'preetham.ganesh2015@gmail.com'


import pickle
import json
import numpy as np


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
