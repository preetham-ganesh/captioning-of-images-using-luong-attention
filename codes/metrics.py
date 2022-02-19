# authors_name = 'Preetham Ganesh'
# project_title = 'Captioning of Images using Attention Mechanism'
# email = 'preetham.ganesh2015@gmail.com'


import sys
import math

import pandas as pd
from nltk.translate import bleu_score
from nltk.translate import meteor_score
from nltk.translate import chrf_score
from nltk.translate import nist_score
from nltk.translate import gleu_score


def calculate_metrics(reference: str,
                      hypothesis: str) -> dict:
    """Using reference (target) sentences, and hypothesis (predict) sentences, calculates metrics such as BLEU, METEOR,
    NIST, CHRF, & GLEU scores.

    Args:
        reference: A string which contains the reference (target) sentence.
        hypothesis: A string which contains the hypothesis (predicted) sentence.

    Returns:
        A dictionary which contains keys as score names and values as scores which are floating point values.
    """
    return {'bleu_score': bleu_score.sentence_bleu([reference], hypothesis),
            'meteor_score': meteor_score.meteor_score([reference], hypothesis, 4),
            'nist_score': nist_score.sentence_nist([reference], hypothesis),
            'chrf_score': chrf_score.sentence_chrf([reference], hypothesis),
            'gleu_score': gleu_score.sentence_gleu([reference], hypothesis)}


def main():
    print()
    attention = ['bahdanau_attention', 'luong_attention']
    model = [1, 2, 3]
    data_split = sys.argv[1]


if __name__ == '__main__':
    main()
    