# authors_name = 'Preetham Ganesh'
# project_title = 'Captioning of Images using Attention Mechanism'
# email = 'preetham.ganesh2015@gmail.com'


import os
import sys

import tensorflow as tf
import tensorflow_datasets as tfds
import logging

from utils import load_pickle_file
from utils import convert_dataset
from utils import check_directory_existence
from utils import save_json_file
from utils import shuffle_slice_dataset
from utils import model_training_validation
from utils import model_testing


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def main():
    print()
    attention = sys.argv[1]
    model_number = int(sys.argv[2])
    train_annotations = load_pickle_file('../data/tokenized_data/annotations', 'train')
    validation_annotations = load_pickle_file('../data/tokenized_data/annotations', 'validation')
    test_annotations = load_pickle_file('../data/tokenized_data/annotations', 'test')
    print('Datasets loaded.')
    print()
    print('No. of captions in the train dataset: {}'.format(len(train_annotations['captions'])))
    print('No. of captions in the validation dataset: {}'.format(len(validation_annotations['captions'])))
    print('No. of captions in the test dataset: {}'.format(len(test_annotations['captions'])))
    print()
    train_image_ids, train_captions = convert_dataset(train_annotations)
    validation_image_ids, validation_captions = convert_dataset(validation_annotations)
    test_image_ids, test_captions = convert_dataset(test_annotations)
    print('Converts dataset into tensors.')
    print()
    print('Shape of train captions: {}'.format(train_captions.shape))
    print('Shape of validation captions: {}'.format(validation_captions.shape))
    print('Shape of test captions: {}'.format(test_captions.shape))
    print()
    batch_size = 64
    target_vocabulary = tfds.deprecated.text.SubwordTextEncoder.load_from_file('../results/utils/captions_tokenizer')
    parameters = {'target_vocab_size': target_vocabulary.vocab_size + 2, 'embedding_size': 512, 'rnn_size': 512,
                  'batch_size': batch_size, 'epochs': 20, 'attention': attention, 'model_number': model_number,
                  'dropout_rate': 0.3, 'start_token_index': target_vocabulary.vocab_size,
                  'train_steps_per_epoch': len(train_image_ids) // batch_size,
                  'validation_steps_per_epoch': len(validation_image_ids) // batch_size,
                  'test_steps_per_epoch': len(test_image_ids) // batch_size}
    directory_path = check_directory_existence('../results/', attention)
    directory_path = check_directory_existence(directory_path, 'model_{}'.format(model_number))
    directory_path = check_directory_existence(directory_path, 'utils')
    save_json_file(parameters, directory_path, 'parameters')
    train_dataset = shuffle_slice_dataset(train_image_ids, train_captions, batch_size)
    validation_dataset = shuffle_slice_dataset(validation_image_ids, validation_captions, batch_size)
    test_dataset = shuffle_slice_dataset(test_image_ids, test_captions, batch_size)
    print('Shuffled & Sliced the datasets.')
    print()
    print('No. of Training steps per epoch: {}'.format(parameters['train_steps_per_epoch']))
    print('No. of Validation steps per epoch: {}'.format(parameters['validation_steps_per_epoch']))
    print('No. of Testing steps: {}'.format(parameters['test_steps_per_epoch']))
    print()
    print('Model Training started.')
    model_training_validation(train_dataset, validation_dataset, parameters)
    print()
    model_testing(test_dataset, parameters)


if __name__ == '__main__':
    main()
