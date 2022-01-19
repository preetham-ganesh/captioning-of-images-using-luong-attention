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


def find_rare_words(words_count: dict) -> dict:
    """Identifies rare words and removes it from the word vocabulary.

    Identifies rare words in the dataset. A word is considered rare if the count is 1 and if the word is not
    classified as digit and if word is a classified as alphabets. The rare words are removed from the words count
    dictionary.

    Args:
        words_count: A dictionary which contains the unique words in the dataset and the number of times it occured in
                     the dataset.

    Returns:
        A dictionary which contains all unique words from the dataset apart from the rare words.
    """
    rare_words = list()
    for word in words_count.keys():
        if words_count[word] == 1 and not word.isdigit() and word.isalpha():
            rare_words.append(word)
    new_words_count = dict()
    for word in words_count.keys():
        if word not in rare_words:
            new_words_count[word] = words_count[word]
    return new_words_count


def converts_rare_words(sentences: list,
                        frequent_words: dict) -> list:
    """Converts all identified rare words in the dataset into a single common token ('@@@').

    Args:
        sentences: Processed captions in the dataset for the current data split.
        frequent_words: A list which contains frequent words in the train dataset.

    Returns:
        An updated list of sentences which contains only the frequent words and the common unknown token ('@@@').
    """
    converted_sentences = list()
    # Iterates across processed captions
    for i in range(len(sentences)):
        current_sentence = sentences[i].split(' ')
        current_sentence_converted = list()
        # Iterates across words / tokens in a sentence
        for j in range(len(current_sentence)):
            # If current word in frequent words then it is appended, else the common unknown token '@@@' is appended to
            # the modified current sentence.
            if current_sentence[j] not in frequent_words.keys():
                current_sentence_converted.append('@@@')
            else:
                current_sentence_converted.append(current_sentence[j])
        current_sentence_converted = ' '.join(current_sentence_converted)
        converted_sentences.append(current_sentence_converted)
    return converted_sentences


def main():
    print()
    processed_train_dataset = pd.read_csv('../data/processed_data/annotations/train.csv')
    print('No. of captions in the processed train dataset: {}'.format(len(processed_train_dataset)))
    processed_train_captions_text = lines_to_text(processed_train_dataset['captions'], '\n')
    print()
    processed_train_captions_letters, processed_train_captions_words = text_stats(processed_train_captions_text)
    print('No. of unique characters in the processed train dataset: {}'.format(len(
        processed_train_captions_letters.keys())))
    print('No. of unique words in the processed train dataset: {}'.format(len(processed_train_captions_words.keys())))
    print()
    processed_train_rare_words = find_rare_words(processed_train_captions_words)
    print('No. of rare words in the processed train dataset: {}'.format(len(processed_train_rare_words)))
    print()


if __name__ == '__main__':
    main()
