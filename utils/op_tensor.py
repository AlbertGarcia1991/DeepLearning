from numpy import ndarray, unique
from typing import Optional, Union
import tensorflow as tf


def op_flatten_input(X: ndarray) -> tf.Tensor:
    """
    Flattens the features of the given input array, keeping the first dimension indicating the
    number of samples.

    Args:
        X: Input array. First dimension must be the number of samples N.

    Returns:
        flat_x: Output flattened array with shape (N, -1)
    """
    out_dims = tf.reduce_prod(X.shape[1:]).numpy()
    flat_x = tf.reshape(X, (-1, out_dims))
    return flat_x


def op_one_hot(y: ndarray, n_classes: Optional[int] = None) -> tf.Tensor:
    """
    Performs one-hot encoding of the given array.

    Args:
        y: Output array encoded with numerical labels.
        n_classes: Number of classes of the output array.

    Returns:
        y_oh: One hot encoded array.
    """
    # TODO: Implement one-hot encoding if original classes are not numerical
    if n_classes is None:
        n_classes = len(unique(y))
    y_oh = tf.one_hot(indices=y, depth=n_classes)
    return y_oh
