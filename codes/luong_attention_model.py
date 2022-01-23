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

    def __init__(self, dense_size: int):
        """Initializes the layers in the instance based on the dense size"""
        super(LuongAttention, self).__init__()
        self.w_a = tf.keras.layers.Dense(dense_size)

