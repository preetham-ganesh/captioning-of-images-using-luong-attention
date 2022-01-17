# authors_name = 'Preetham Ganesh'
# project_title = 'Captioning of Images using Attention Mechanism'
# email = 'preetham.ganesh2015@gmail.com'


import tensorflow as tf


class Encoder(tf.keras.Model):
    """Encodes extracted features for future interpretation.

    The features extracted from the images using the pre-trained InceptionV3 model are encoded to the embedding size by
    passing it into a fully connected layer. The encoded output will be passed on to the Bahdanau Attention layer for
    future interpretation.


    Args:
        dense_layer: Fully connected layer which encodes the extracted features for future interpretation.
        dropout_layer: Dropout layer which prevents the model from overfitting on the training dataset.
    """

    def __init__(self, embedding_size: int,
                 dropout_rate: float) -> None:
        """Initializes the instance based on the embedding size, and dropout_rate.

        Args:
            embedding_size: No. of units in the fully connected layer.
            dropout_rate: Rate at which the number of input units should be dropped.
        """
        self.dense_layer = tf.keras.layers.Dense(embedding_size)
        self.dropout_layer = tf.keras.layers.Dropout(rate=dropout_rate)