# authors_name = 'Preetham Ganesh'
# project_title = 'Captioning of Images using Attention Mechanism'
# email = 'preetham.ganesh2015@gmail.com'


import pandas as pd
from collections import Counter


def lines_to_text(sentences_list: list,
                  separator: str) -> str:
    """Converts sentences to text.

    Converts list of sentences into a single string by using separator as the joining criteria.

    Args:
        sentences_list: List that contains sentences which should be converted into text.
        separator: String by which the list of sentences should be concatenated.

    Returns:
        A single string which contains all the sentences with separator as the joining criteria.
    """
    sentences_text = ''
    # Iterates across list of sentences.
    for i in range(len(sentences_list)):
        if i == len(sentences_list) - 1:
            sentences_text += str(sentences_list[i])
        else:
            sentences_text += str(sentences_list[i]) + separator
    return sentences_text


def text_stats(processed_captions_text: str) -> tuple:
    """Provides stats about the text such as number of unique characters and words.

    Args:
        processed_captions_text: A single string which contains all the sentences in the dataset.

    Returns:
        A tuple which contains unique characters and unique words from the dataset.
    """
    unique_letters = Counter(processed_captions_text)
    unique_words = Counter(processed_captions_text.split(' '))
    return unique_letters, unique_words


def find_rare_words(words_count: dict) -> list:
    """Identifies rare words in the dataset. A word is considered rare if the count is 1 or if the word is not
    classified as digit or if word is a classified as alphabets.

    Args:
        words_count: A dictionary which contains the unique words in the dataset and the number of times it occured in
                     the dataset.

    Returns:
        A list which contains rare words in the dataset.
    """
    rare_words = list()
    for word in words_count.keys():
        if words_count[word] == 1 or not word.isdigit() or word.isalpha():
            rare_words.append(word)
    return rare_words


def main():
    print()
    processed_train_dataset = pd.read_csv('../data/processed_data/annotations/train.csv')
    print('No. of captions in the processed train dataset: {}'.format(len(processed_train_dataset)))
    processed_train_captions_text = lines_to_text(processed_train_dataset['captions'], '\n')
    print()
    processed_train_captions_letters, processed_train_captions_words = text_stats(processed_train_captions_text)





if __name__ == '__main__':
    main()
