# authors_name = 'Preetham Ganesh'
# project_title = 'Captioning of Images using Attention Mechanism'
# email = 'preetham.ganesh2015@gmail.com'


import tensorflow as tf


class Encoder(tf.keras.Model):
    """Encodes features extracts from images for future interpretation.

    The features extracted from the images using the pre-trained InceptionV3 model are encoded to the embedding size by
    passing it into a fully connected layer. The encoded output will be passed on to the Bahdanau Attention layer for
    future interpretation.

    Args:
        dense_layer: Fully connected layer which encodes the extracted features for future interpretation.
        dropout_layer: Dropout layer which prevents the model from overfitting on the training dataset.
    """

    def __init__(self, embedding_size: int,
                 dropout_rate: float) -> None:
        """Initializes the layers in the instance based on the embedding size, and dropout_rate."""
        super(Encoder, self).__init__()
        self.dense_layer = tf.keras.layers.Dense(embedding_size)
        self.dropout_layer = tf.keras.layers.Dropout(rate=dropout_rate)

    def call(self, x: tf.Tensor,
             training: bool) -> tf.Tensor:
        """Input tensor is passed through the layers in the encoder model."""
        x = self.dense_layer(x)
        x = self.dropout_layer(x, training=training)
        return x


class BahdanauAttention(tf.keras.Model):
    """A local attention model which uses the input and previous timestep's output to predict the output for the current
    timestep.

    Args:
        w_1: Weights for the Encoder's output
        w_2: Weights for the Encoder's hidden state h.
        w_3: Weights for the Encoder's hidden state c.
        v: Final layer which sums output from w_1, w_2, & w_3.
    """

    def __init__(self, dense_size: int):
        """Initializes the layers in the instance based on the dense size"""
        super(BahdanauAttention, self).__init__()
        self.w_1 = tf.keras.layers.Dense(dense_size)
        self.w_2 = tf.keras.layers.Dense(dense_size)
        self.w_3 = tf.keras.layers.Dense(dense_size)
        self.v = tf.keras.layers.Dense(1)

    def call(self, encoder_out: tf.Tensor,
             hidden_state_h: tf.Tensor,
             hidden_state_c: tf.Tensor):
        """Encoder output, and hidden states are passed the through the layers in the Bahdanau Attention model."""
        # Inserts a length 1 at axis 1 in the hidden states.
        hidden_state_h_time = tf.expand_dims(hidden_state_h, 1)
        hidden_state_c_time = tf.expand_dims(hidden_state_c, 1)
        # Provides un-normalized score for each feature.
        attention_hidden_layer = self.v(tf.nn.tanh(self.w_1(encoder_out) + self.w_2(hidden_state_h_time) +
                                                   self.w_3(hidden_state_c_time)))
        # Uses softmax on output from attention_hidden_layer to predict the output.
        attention_out = tf.nn.softmax(attention_hidden_layer, axis=1)
        context_vector = attention_out * encoder_out
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector