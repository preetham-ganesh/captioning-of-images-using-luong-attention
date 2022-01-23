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

