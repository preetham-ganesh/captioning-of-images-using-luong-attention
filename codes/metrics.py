# authors_name = 'Preetham Ganesh'
# project_title = 'Captioning of Images using Attention Mechanism'
# email = 'preetham.ganesh2015@gmail.com'


import pandas as pd
import numpy as np
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


def calculate_metrics_mean(current_data_split_mean_metrics: pd.DataFrame,
                           current_data_split_model_metrics: pd.DataFrame,
                           metric_features: list,
                           attention: str,
                           model: int) -> pd.DataFrame:
    """Calculates the mean of the metrics computed for every target and predicted sentence in the current data split.

    Args:
        current_data_split_mean_metrics: A dataframe containing the mean metrics for all model configurations.
        current_data_split_model_metrics: A dataframe containing the metrics for the current configuration.
        metric_features: A list containing the acronyms of the metrics used for evaluating the trained models.
        attention: A string which contains the name of the current attention model.
        model: An integer which contains the number of the current model.

    Returns:
        An updated dataframe containing the mean of all metrics for all model configurations including the current
        configuration.
    """
    # Calculates mean of all the metrics computed for all sentences in the current data split.
    current_data_split_model_mean_metrics = {
        metric_features[i]: round(np.mean(current_data_split_model_metrics[metric_features[i]]), 6) for i in
        range(len(metric_features))}
    current_data_split_model_mean_metrics['attention'] = attention
    current_data_split_model_mean_metrics['model'] = model
    # Appends the current model configuration's mean of metrics to the current_data_split_mean_metrics
    current_data_split_mean_metrics = current_data_split_mean_metrics.append(current_data_split_model_mean_metrics,
                                                                             ignore_index=True)
    return current_data_split_mean_metrics


def per_model_calculate_metrics(data_split: list,
                                attention: list,
                                model: list) -> None:
    """Calculates the mean of the metrics for every target and predicted sentence in all data splits.

    Args:
        data_split: A list which contains data split used for evaluating the models.
        attention: A list which contains the names of the attention models used.
        model: A list which contains the number of the models used.

    Returns:
        None.
    """
    metric_features = ['bleu_score', 'meteor_score', 'nist_score', 'chrf_score', 'gleu_score']
    # Iterates across the data splits used for evaluating the models.
    for i in range(len(data_split)):
        # Creates an empty dataframe for storing the mean of all metrics for the current data split.
        current_data_split_mean_metrics = pd.DataFrame(columns=['attention', 'model'] + metric_features)
        # Iterates across the different model configurations used for developing the models.
        for j in range(len(attention)):
            for k in range(len(model)):
                current_data_split_model_predictions = pd.read_csv(
                    '../results/{}/model_{}/predictions/{}.csv'.format(attention[j], model[k], data_split[i]))
                # Creates an empty dataframe for storing metrics per sentence.
                current_data_split_model_metrics = pd.DataFrame(columns=metric_features)
                # Iterates across the sentences predicted by the current model configuration.
                for m in range(len(current_data_split_mean_metrics)):
                    current_target_caption = current_data_split_model_predictions.iloc[m]['target_caption']
                    current_predicted_caption = current_data_split_model_predictions.iloc[m]['predicted_caption']
                    # Calculates metrics for current target and predicted captions.
                    current_index_calculated_metrics = calculate_metrics(current_target_caption,
                                                                         current_predicted_caption)
                    current_data_split_model_metrics = current_data_split_model_metrics.append(
                        current_index_calculated_metrics, ignore_index=True)
                # Computes mean of metrics for all sentences in the current data split.
                current_data_split_mean_metrics = calculate_metrics_mean(current_data_split_mean_metrics,
                                                                         current_data_split_model_metrics,
                                                                         metric_features, attention[j], model[k])
        print('Metrics computed for {} data.'.format(data_split[i]))
        print()
        print(current_data_split_mean_metrics)
        current_data_split_mean_metrics.to_csv('../results/utils/{}.csv'.format(data_split[i]), index=False)
        print()


def main():
    print()
    attention = ['bahdanau_attention', 'luong_attention']
    model = [1, 2, 3]
    data_split = ['validation', 'test']
    per_model_calculate_metrics(data_split, attention, model)


if __name__ == '__main__':
    main()
