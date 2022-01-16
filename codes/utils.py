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
            Loaded JSON file which contains the image file names and captions in the current split of the dataset.
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


