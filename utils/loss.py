import tensorflow as tf


def loss_rmse(y_pred: tf.Tensor, y_true: tf.Tensor) -> float:
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


def loss_mse(y_pred: tf.Tensor, y_true: tf.Tensor) -> float:
    """
    Compute root-mean squared error loss.

    Args:
        y_pred: Output array containing the predicted values.
        y_true: Output array containing the (ground) true values.

    Returns:
        loss: Computer loss value.
    """
    loss = tf.reduce_mean(input_tensor=tf.square(x=y_true - y_pred))
    return loss


def loss_sparse_cross_entropy(y_pred: tf.Tensor, y_true: tf.Tensor) -> float:
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


def loss_cross_entropy(y_pred: tf.Tensor, y_true: tf.Tensor) -> float:
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
