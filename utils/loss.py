import tensorflow as tf


def loss_rmse(y_pred: tf.Tensor, y_true: tf.Tensor) -> tf.Tensor:
    """
    Compute root-mean squared error loss.

    Args:
        y_pred: Output array containing the predicted values.
        y_true: Output array containing the (ground) true values.

    Returns:
        loss: Computer loss value.
    """
    loss = tf.sqrt(x=tf.reduce_mean(input_tensor=tf.square(x=y_true - y_pred)))
    return loss


def loss_mse(y_pred: tf.Tensor, y_true: tf.Tensor) -> tf.Tensor:
    """
    Compute mean squared error loss.

    Args:
        y_pred: Output array containing the predicted values.
        y_true: Output array containing the (ground) true values.

    Returns:
        loss: Computer loss value.
    """
    loss = tf.reduce_mean(input_tensor=tf.square(x=y_true - y_pred))
    return loss


def loss_mae(y_pred: tf.Tensor, y_true: tf.Tensor) -> tf.Tensor:
    """
    Compute mean absolute error loss. More robust to outliers than MSE and RMSE.

    Args:
        y_pred: Output array containing the predicted values.
        y_true: Output array containing the (ground) true values.

    Returns:
        loss: Computer loss value.
    """
    loss = tf.reduce_mean(input_tensor=tf.abs(x=y_true - y_pred))
    return loss


def loss_huber(y_pred: tf.Tensor, y_true: tf.Tensor) -> tf.Tensor:
    """
    Compute Huber loss. Less sensitive to outliers than MSE, but still maintains some of its desirable properties.

    Args:
        y_pred: Output array containing the predicted values.
        y_true: Output array containing the (ground) true values.

    Returns:
        loss: Computer loss value.
    """
    error = y_true - y_pred
    is_small_error = tf.abs(error) <= delta
    quadratic_loss = 0.5 * tf.square(error)
    linear_loss = delta * (tf.abs(error) - 0.5 * delta)
    loss = tf.where(is_small_error, quadratic_loss, linear_loss)
    return loss


def loss_log_cosh(y_pred: tf.Tensor, y_true: tf.Tensor) -> tf.Tensor:
    """
    Compute Log-Cosh loss.  It is smooth and has some robustness to outliers similar to Huber loss.

    Args:
        y_pred: Output array containing the predicted values.
        y_true: Output array containing the (ground) true values.

    Returns:
        loss: Computer loss value.
    """
    loss = tf.reduce_mean(tf.cosh(y_true - y_pred) - 1)
    return loss


def loss_sparse_cross_entropy(y_pred: tf.Tensor, y_true: tf.Tensor) -> tf.Tensor:
    """
    Compute cross-entropy loss with a sparse operation. This means that labels must have length [batch_size] and dtype
    of int32 or int64.

    Args:
        y_pred: Output array containing the predicted values.
        y_true: Output array containing the (ground) true values.

    Returns:
        loss: Computer loss value.
    """
    #
    sparse_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    loss = tf.reduce_mean(input_tensor=sparse_ce)
    return loss


def loss_cross_entropy(y_pred: tf.Tensor, y_true: tf.Tensor) -> tf.Tensor:
    """
    Compute cross-entropy loss with a non-sparse operation. This means that labels must have length
    [batch_size, n_classes] and dtype of float32 or float64.

    Args:
        y_pred: Output array containing the predicted values.
        y_true: Output array containing the (ground) true values.

    Returns:
        loss: Computer loss value.
    """
    # Compute cross-entropy loss with a sparse operation
    sparse_ce = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    loss = tf.reduce_mean(input_tensor=sparse_ce)
    return loss


def loss_binary_cross_entropy(y_true: tf.Tensor, y_pred: tf.Tensor, eps: float = 1e-7) -> tf.Tensor:
    """
    Compute binary cross-entropy loss.

    Args:
        y_true: A tensor of true labels, shape (batch_size, 1).
        y_pred: A tensor of predicted probabilities, shape (batch_size, 1).
        eps: A small constant for numerical stability.

    Returns:
        loss: A scalar representing the binary cross-entropy loss.
    """
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.clip_by_value(y_pred, clip_value_min=eps, clip_value_max=1.0 - eps)
    loss = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
    return tf.reduce_mean(loss)


def loss_kl_divergence(y_pred: tf.Tensor, y_true: tf.Tensor, eps: float = 1e-8) -> tf.Tensor:
    """
    Compute the Kullback-Leibler divergence, which measures the difference between two probability distributions.
    Commonly used in autoencoders and variational autoencoders.

    Args:
        y_pred: Output array containing the predicted values.
        y_true: Output array containing the (ground) true values.
        eps: Clipping tolerance to avoid pure zeros on the operations.

    Returns:
        loss: Computer loss value.
    """
    # Compute cross-entropy loss with a sparse operation
    y_true = tf.clip_by_value(y_true, eps, 1)
    y_pred = tf.clip_by_value(y_pred, eps, 1)
    loss = tf.reduce_sum(y_true * tf.math.log(y_true / y_pred), axis=-1)
    return loss


def loss_triplet(anchor: tf.Tensor, positive: tf.Tensor, negative: tf.Tensor, margin: float = 1.0) -> tf.Tensor:
    """
    Compute the triplet loss. It compares a baseline input to positive input and a negative input. The distance between
    the baseline input and the positive input is reduced to the minimum, while the distance between the baseline input
    and the negative input is maximized. Is commonly used in anomaly-detection applications and for similarity learning
    tasks, it is designed to learn a distance metric between quell_gestures points. It is often used for tasks like face
    recognition and image retrieval.

    Args:
        anchor: Reference input or y_true.
        positive: Matching input or y_pred well-classified example.
        negative: Non-matching input or y_pred misclassified example.
        margin: Value used to keep negative samples far apart.

    Returns:
        loss: Computer loss value.
    """
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    loss = tf.maximum(pos_dist - neg_dist + margin, 0)
    return loss


def loss_contrastive(y_true: tf.Tensor, y_pred: tf.Tensor, margin: float = 1.0) -> tf.Tensor:
    """
    Compute the contrastive loss. It takes the output of the network for a positive example and calculates its distance
    to an example of the same class and contrasts that with the distance to negative examples. Mainly used on siamese
    networks and unsupervised algorithms. Also used for similarity learning tasks, it encourages similar quell_gestures points to
    have similar representations and dissimilar quell_gestures points to have different representations.

    Args:
        y_pred: Output array containing the predicted values.
        y_true: Output array containing the (ground) true values.
        margin: Defines a radius around the embedding space of a sample so that dissimilar pairs of samples only
        contribute to the contrastive loss function if the squared_difference is within the margin.

    Returns:
        loss: Computer loss value.
    """
    y_true = tf.cast(y_true, y_pred.dtype)
    squared_difference = tf.square(y_pred)
    loss = tf.reduce_mean(y_true * squared_difference + (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0)))
    return loss


def loss_focal(
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        gamma: float = 2.0,
        alpha: float = 0.25,
        eps: float = 1e-8
) -> tf.Tensor:
    """
    Compute the focal loss. Is used to address the issue of the class imbalance problem. A modulation term is applied to
    cross-entropy loss function, making it more efficient and easy to learn for hard examples.

    Args:
        y_pred: Output array containing the predicted values.
        y_true: Output array containing the (ground) true values.
        gamma: Relaxation parameter. The bigger it is, the more importance is given to misclassified examples and very
            loss will be propagated from easy examples.
        alpha: Controls the degree to which the loss is down-weighted for well-classified examples. Is typically
            set such that the majority class has a lower weighting than the minority classes. Must be between 0 and 1.
        eps: Tolerance to avoid zeroes inside the log operations.

    Returns:
        loss: Computer loss value.
    """
    y_true = tf.cast(y_true, y_pred.dtype)
    y_pred = tf.clip_by_value(y_pred, eps, 1 - eps)
    ce_loss = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
    weight = y_true * tf.pow(1 - y_pred, gamma) + (1 - y_true) * tf.pow(y_pred, gamma)
    loss = alpha * tf.reduce_sum(weight * ce_loss, axis=-1)
    return loss
