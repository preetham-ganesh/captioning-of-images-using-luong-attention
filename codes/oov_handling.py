# authors_name = 'Preetham Ganesh'
# project_title = 'Captioning of Images using Attention Mechanism'
# email = 'preetham.ganesh2015@gmail.com'


import pandas as pd
from collections import Counter


def lines_to_text(sentences_list: list,
                  separator: str):
    """Converts sentences to text.

    Converts list of sentences into a single string by using separator as the joining criteria.

    Args:
        sentences_list: List that contains sentences which should be converted into text.
        separator: String by which the list of sentences should be concatenated.

    Returns:
        A single string which contains all the sentences with separator as the joining criteria.
    """
    sentences_text = ''
    for i in range(len(sentences_list)):
        if i == len(sentences_list) - 1:
            sentences_text += str(sentences_list[i])
        else:
            sentences_text += str(sentences_list[i]) + separator
    return sentences_text


def main():
    print()
    processed_train_dataset = pd.read_csv('../data/processed_data/annotations/train.csv')
    processed_train_captions_text =



if __name__ == '__main__':
    main()
