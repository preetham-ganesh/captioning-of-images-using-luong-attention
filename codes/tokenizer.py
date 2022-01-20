# authors_name = 'Preetham Ganesh'
# project_title = 'Captioning of Images using Attention Mechanism'
# email = 'preetham.ganesh2015@gmail.com'


import tensorflow_datasets as tfds
import pandas as pd
from utils import save_pickle_file


def per_dataset_tokenizer(captions_tokenizer: tfds.deprecated.text.SubwordTextEncoder,
                          dataset: pd.DataFrame,
                          data_split: str) -> None:
    """Tokenizes captions in the dataset for the current data split suing the trained captions tokenizer.

    Args:
        captions_tokenizer: TFDS tokenizer trained on the captions from the train dataset.
        dataset: A pandas dataframe which contains image_ids and captions for the current data split.
        data_split: Split the annotations (captions) belong to i.e., train, val or test.

    Returns:
        None.
    """
    # Encodes all sentences in the dataset by adding a start token and end token in every sentence.
    tokenized_captions = [[captions_tokenizer.vocab_size] + captions_tokenizer.encode(i) +
                          [captions_tokenizer.vocab_size] for i in list(dataset['captions'])]
    tokenized_data = {'image_ids': list(dataset['image_ids']), 'captions': tokenized_captions}
    save_pickle_file(tokenized_data, '../data/tokenized_data/annotations/', data_split)


def tokenizer_train_save(train_dataset: pd.DataFrame,
                         vocabulary_size: int) -> tfds.deprecated.text.SubwordTextEncoder:
    """Trains TFDS tokenizer on the train dataset and saves trained tokenizer.

    Args:
        train_dataset: A pandas dataframe which contains image_ids and captions in the train dataset.
        vocabulary_size: Approximate size by which TFDS tokenizer should be trained.

    Returns:
        A TFDS tokenizer trained on the train dataset.
    """
    captions_tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus((i for i in list(
        train_dataset['captions'])), target_vocab_size=vocabulary_size)
    print('Captions tokenizer trained.')
    captions_tokenizer.save_to_file('../results/utils/captions_tokenizer')
    print('Captions tokenizer saved.')
    return captions_tokenizer


def main():
    print()
    oov_handled_train_dataset = pd.read_csv('../data/oov_handled_data/annotations/train.csv')
    oov_handled_validation_dataset = pd.read_csv('../data/oov_handled_data/annotations/validation.csv')
    oov_handled_test_dataset = pd.read_csv('../data/oov_handled_data/annotations/test.csv')
    vocabulary_size = 2 ** 11
    captions_tokenizer = tokenizer_train_save(oov_handled_train_dataset, vocabulary_size)
    print()
    per_dataset_tokenizer(captions_tokenizer, oov_handled_train_dataset, 'train')
    print('Tokenized captions in train dataset and saved successfully.')
    per_dataset_tokenizer(captions_tokenizer, oov_handled_validation_dataset, 'validation')
    print('Tokenized captions in validation dataset and saved successfully.')
    per_dataset_tokenizer(captions_tokenizer, oov_handled_test_dataset, 'test')
    print('Tokenized captions in test dataset and saved successfully.')
    print()


if __name__ == '__main__':
    main()