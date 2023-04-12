from typing import Tuple
import tensorflow as tf


def init_weights_xavier(shape: Tuple[int, int]) -> tf.Tensor:
    """
    Initialization of layer weights using Xavier algorithm.

    Args:
        shape: Shape of the layer as a tuple (in_size, out_size)

    Returns:
        weight_vals: Computed weights.
    """
    in_dims, out_dims = shape
    xavier_lim = tf.sqrt(x=6.) / tf.sqrt(x=tf.cast(x=in_dims + out_dims, dtype=tf.float32))
    weight_vals = tf.random.uniform(shape=shape, minval=-xavier_lim, maxval=xavier_lim, seed=13)
    return weight_vals
