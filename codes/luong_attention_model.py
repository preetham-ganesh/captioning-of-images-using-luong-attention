# authors_name = 'Preetham Ganesh'
# project_title = 'Captioning of Images using Attention Mechanism'
# email = 'preetham.ganesh2015@gmail.com'


import tensorflow as tf


class LuongAttention(tf.keras.Model):
    """A global attention model which uses the output from final RNN layer in the decoder and the encoder output to
    predict the output for the current timestep.

    Args:
        w_a: Weights for the Encoder's output.
    """

    def __init__(self, dense_size: int) -> None:
        """Initializes the layers in the instance based on the dense size."""
        super(LuongAttention, self).__init__()
        self.w_a = tf.keras.layers.Dense(dense_size)

    def call(self, encoder_out: tf.Tensor,
             decoder_out: tf.Tensor) -> tf.Tensor:
        """Encoder output, and decoder output are passed through the layers in the Luong Attention model."""
        attention_hidden_layer = tf.matmul(decoder_out, self.w_a(encoder_out), transpose_b=True)
        attention_out = tf.nn.softmax(attention_hidden_layer, axis=2)
        context_vector = tf.matmul(attention_out, encoder_out)
        return context_vector


class Decoder1(tf.keras.Model):
    """Decodes the features encoded using the Encoder model and predicts output for the current timestep using Luong
    Attention.

    Args:
        attention_layer: Luong attention model which is used to emphasize the important features at different
                         timesteps.
        embedding_layer: Converts indexes from target vocabulary into dense vectors of fixed size.
        rnn_layer: A Long Short-Term Memory layer used to learn dependencies in the given sequence.
        dense_layer_1: Fully connected layer which encodes output sequence from the rnn layer.
        dense_layer_2: Fully connected layer which encodes output sequence from the dense_layer_1 to the target vocab
                       size.
        dropout_layer: Dropout layer which prevents the model from overfitting on the training dataset.
    """

    def __init__(self, embedding_size: int,
                 rnn_size: int,
                 dropout_rate: float,
                 target_vocab_size: int) -> None:
        """Initializes the layers in the instance based on the embedding_size, rnn_size, dropout_rate, and
        target_vocab_size."""
        self.attention_layer = LuongAttention(rnn_size)
        self.embedding_layer = tf.keras.layers.Embedding(target_vocab_size, embedding_size)
        self.rnn_layer = tf.keras.layers.LSTM(rnn_size, return_state=True, return_sequences=True)
        self.dense_layer_1 = tf.keras.layers.Dense(rnn_size, activation='tanh')
        self.dropout_layer = tf.keras.layers.Dropout(rate=dropout_rate)
        self.dense_layer_2 = tf.keras.layers.Dense(target_vocab_size)

