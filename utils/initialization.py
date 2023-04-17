from typing import Tuple

import tensorflow as tf

"""
Weight initialization methods overview and selection rule of thumb:

Xavier (Glorot) Initialization:
    - Pros: Works well for layers with Sigmoid or Tanh activation functions; balances variance of input and output.
    - Cons: Not suitable for ReLU activations.
    - When to use: Deep networks with Sigmoid, Tanh, or Softmax activations.

He Initialization:
    - Pros: Works well for layers with ReLU activations; prevents vanishing gradients.
    - Cons: Not suitable for Sigmoid or Tanh activations.
    - When to use: Deep networks with ReLU or leaky ReLU activations.

LeCun Initialization:
    - Pros: Works well for layers with SELU activations; maintains self-normalizing property.
    - Cons: Not suitable for other activation functions.
    - When to use: Deep networks with SELU activations.

Zero Initialization:
    - Pros: Simple and easy to implement.
    - Cons: Can cause vanishing or exploding gradients; not suitable for deep networks.
    - When to use: For b initialization; not recommended for a initialization.

One Initialization:
    - Pros: Simple and easy to implement.
    - Cons: Can cause vanishing or exploding gradients; not suitable for deep networks.
    - When to use: Rarely used; not recommended for most cases.

Uniform Initialization:
    - Pros: Works well when the initialization range is set appropriately.
    - Cons: Can cause vanishing or exploding gradients if the range is not set properly.
    - When to use: For shallow networks or as a baseline initialization method.

Normal Initialization:
    - Pros: Works well when the mean and standard deviation are set appropriately.
    - Cons: Can cause vanishing or exploding gradients if the parameters are not set properly.
    - When to use: For shallow networks or as a baseline initialization method; can also be used in deeper networks 
        if the activation functions and architecture allow for it.
    
Truncated Normal Initialization:
    - Pros: Similar to normal initialization but avoids extreme values, reducing the chances of vanishing or exploding 
        gradients.
    - Cons: Can still cause vanishing or exploding gradients if the parameters are not set properly.
    - When to use: As an alternative to normal initialization when you want to limit the range of initialized values.

Orthogonal Initialization:
    - Pros: Preserves the norm of the input, which can help with gradient flow and convergence.
    - Cons: Only applicable to square matrices or layers with an equal number of input and output units.
    - When to use: For recurrent neural networks (RNNs) or when the input and output dimensions are equal.

Identity Initialization:
    - Pros: Preserves the identity of the input, which can help with gradient flow and convergence.
    - Cons: Only applicable to square matrices or layers with an equal number of input and output units.
    - When to use: For RNNs or when the input and output dimensions are equal, and you want the layer to initially 
        behave like an identity mapping.
"""


def init_weights_xavier(shape: Tuple[int, int]) -> tf.Tensor:
    """
    Initialization of layer weights using Xavier algorithm (aka Glorot Normal).

    Args:
        shape: Shape of the layer as a tuple (in_size, out_size)

    Returns:
        weight_vals: Computed weights.
    """
    in_dims, out_dims = shape
    xavier_lim = tf.sqrt(x=6.) / tf.sqrt(x=tf.cast(x=in_dims + out_dims, dtype=tf.float32))
    weight_vals = tf.random.uniform(shape=shape, minval=-xavier_lim, maxval=xavier_lim, seed=13)
    return weight_vals


def init_weights_he(shape: Tuple[int, int]) -> tf.Tensor:
    """
    Initialization of layer weights using He algorithm.

    Args:
        shape: Shape of the layer as a tuple (in_size, out_size)

    Returns:
        weight_vals: Computed weights.
    """
    in_dims, _ = shape
    std_dev = tf.sqrt(x=2 / tf.cast(x=in_dims, dtype=tf.float32))
    weight_vals = tf.random.normal(shape=shape, mean=0., stddev=std_dev, seed=13)
    return weight_vals


def init_weights_lecun(shape: Tuple[int, int]) -> tf.Tensor:
    """
    Initialization of layer weights using LeCun algorithm.

    Args:
        shape: Shape of the layer as a tuple (in_size, out_size)

    Returns:
        weight_vals: Computed weights.
    """
    in_dims, _ = shape
    std_dev = tf.sqrt(x=1 / tf.cast(x=in_dims, dtype=tf.float32))
    weight_vals = tf.random.normal(shape=shape, mean=0., stddev=std_dev, seed=13)
    return weight_vals


def init_weights_zero(shape: Tuple[int, int]) -> tf.Tensor:
    """
    Initialization of layer weights setting all values to 0.

    Args:
        shape: Shape of the layer as a tuple (in_size, out_size)

    Returns:
        weight_vals: Computed weights.
    """
    weight_vals = tf.zeros(shape=shape, dtype=tf.float32)
    return weight_vals


def init_weights_one(shape: Tuple[int, int]) -> tf.Tensor:
    """
    Initialization of layer weights setting all values to 1.

    Args:
        shape: Shape of the layer as a tuple (in_size, out_size)

    Returns:
        weight_vals: Computed weights.
    """
    weight_vals = tf.ones(shape=shape, dtype=tf.float32)
    return weight_vals


def init_weights_uniform(shape: Tuple[int, int], min_val: float = -0.05, max_val: float = 0.05) -> tf.Tensor:
    """
    Initialization of layer weights setting all values to follow a Uniform distribution.

    Args:
        shape: Shape of the layer as a tuple (in_size, out_size)
        min_val: Minimum value of the uniform distribution.
        max_val: Maximum value of the uniform distribution.

    Returns:
        weight_vals: Computed weights.
    """
    weight_vals = tf.random.uniform(shape=shape, minval=min_val, maxval=max_val, seed=13)
    return weight_vals


def init_weights_normal(shape: Tuple[int, int], mean: float = 0., stddev: float = 0.05) -> tf.Tensor:
    """
    Initialization of layer weights setting all values to follow a Normal (Gaussian) distribution.

    Args:
        shape: Shape of the layer as a tuple (in_size, out_size).
        mean: Mean value of the Normal distribution.
        stddev: Standard deviation of the Normal distribution.

    Returns:
        weight_vals: Computed weights.
    """
    weight_vals = tf.random.normal(shape=shape, mean=mean, stddev=stddev, seed=13)
    return weight_vals


def init_weights_truncated_normal(shape: Tuple[int, int], mean: float = 0., stddev: float = 0.05) -> tf.Tensor:
    """
    Initialization of layer weights setting all values to follow a truncated Normal distribution. It will no generate
    values bigger than 2 standard deviations from the mean (truncated).

    Args:
        shape: Shape of the layer as a tuple (in_size, out_size).
        mean: Mean value of the Normal distribution.
        stddev: Standard deviation of the Normal distribution.

    Returns:
        weight_vals: Computed weights.
    """
    weight_vals = tf.random.truncated_normal(shape=shape, mean=mean, stddev=stddev, seed=13)
    return weight_vals


def init_weights_orthogonal(shape: Tuple[int, int], gain: float = 1.0) -> tf.Tensor:
    """
    Initialization of layer weights setting all values to follow a Normal orthogonal distribution. This ensures that
    gradients will not explode or vanish. It is normal because it generates a QR matrix with values drawn from a Normal
    distribution.

    Args:
        shape: Shape of the layer as a tuple (in_size, out_size).
        gain: Multiplicative factor to apply to the orthogonal matrix.

    Returns:
        weight_vals: Computed weights.
    """
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = tf.random.normal(shape=flat_shape)
    q, r = tf.linalg.qr(a)
    d = tf.linalg.diag_part(r)
    ph = tf.sign(d)
    q *= ph
    weight_vals = gain * tf.reshape(q, shape)
    return weight_vals


def init_weights_identity(shape: Tuple[int, int], gain: float = 1.0) -> tf.Tensor:
    """
    Initialization of layer weights setting all values to follow a diagonal matrix.

    Args:
        shape: Shape of the layer as a tuple (in_size, out_size).
        gain: Multiplicative factor to apply to the orthogonal matrix.

    Returns:
        weight_vals: Computed weights.
    """
    if shape[0] != shape[1]:
        raise ValueError("Identity initialization requires a square a matrix.")
    weight_vals = gain * tf.eye(num_rows=shape[0], num_columns=shape[1], dtype=tf.float32)
    return weight_vals
