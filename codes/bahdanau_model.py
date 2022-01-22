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


class BahdanauAttention(tf.keras.layers.Layer):
    """A local attention model which uses the input and previous timestep's output to predict the output for the current
    timestep.

    Args:
        w_1: Weights for the Encoder's output
        w_2: Weights for the Encoder's hidden state h.
        w_3: Weights for the Encoder's hidden state c.
        v: Final layer which sums output from w_1, w_2, & w_3.
    """

    def __init__(self, dense_size: int) -> None:
        """Initializes the layers in the instance based on the dense size"""
        super(BahdanauAttention, self).__init__()
        self.w_1 = tf.keras.layers.Dense(dense_size)
        self.w_2 = tf.keras.layers.Dense(dense_size)
        self.w_3 = tf.keras.layers.Dense(dense_size)
        self.v = tf.keras.layers.Dense(1)

    def call(self, encoder_out: tf.Tensor,
             hidden_state_h: tf.Tensor,
             hidden_state_c: tf.Tensor) -> tf.Tensor:
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


class Decoder1(tf.keras.Model):
    """Decodes the features encoded using the Encoder model and predicts output for the current timestep using Bahdanau
    Attention.

    Args:
        attention_layer: Bahdanau attention model which is used to emphasize the important features at different
                         timesteps.
        embedding_layer: Converts indexes from target vocabulary into dense vectors of fixed size.
        rnn_layer: An Long Short-Term Memory layer used to learn dependencies in the given sequence.
        dense_layer_1: Fully connected layer which encodes the output sequence from the rnn layer.
        dropout_layer: Dropout layer which prevents the model from overfitting on the training dataset.
        dense_layer_2: Fully connected layer which encodes the output sequence from the dropout layer.
    """

    def __init__(self, embedding_size: int,
                 rnn_size: int,
                 target_vocab_size: int,
                 dropout_rate: float) -> None:
        """Initializes the layers in the instance based on the embedding size, rnn_size, target_vocab_size, and
        dropout_rate."""
        super(Decoder1, self).__init__()
        self.attention_layer = BahdanauAttention(rnn_size)
        self.embedding_layer = tf.keras.layers.Embedding(target_vocab_size, embedding_size)
        self.rnn_layer = tf.keras.layers.LSTM(rnn_size, return_state=True, return_sequences=True)
        self.dense_layer_1 = tf.keras.layers.Dense(rnn_size, activation='tanh')
        self.dropout_layer = tf.keras.layers.Dropout(rate=dropout_rate)
        self.dense_layer_2 = tf.keras.layers.Dense(target_vocab_size)

    def call(self, x: tf.Tensor,
             encoder_out: tf.Tensor,
             hidden_states: list,
             training: bool) -> tuple:
        """Input for current timestep, encoder output, and hidden states are passed through the layers in the decoder
        model"""
        context_vector = self.attention_layer(encoder_out, hidden_states[0], hidden_states[1])
        x = self.embedding_layer(x)
        # Concatenates context vector with embedding output.
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        x, hidden_state_h, hidden_state_c = self.rnn_layer(x)
        x = self.dropout_layer(x, training=training)
        # Reshape current output to (batch_size * max_length, hidden_size).
        x = tf.reshape(x, (-1, x.shape[2]))
        x = self.dense_layer_2(x)
        return x, [hidden_state_h, hidden_state_c]
