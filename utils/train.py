import tensorflow as tf
from utils.model import ModelBase
from utils.optimizer import OptimizerBase
from typing import Callable, Tuple, List, Union


@tf.function
def _train_step(
        X_batch: tf.Tensor,
        y_batch: tf.Tensor,
        loss: Callable,
        acc: Callable,
        model: Union[tf.Module, ModelBase],
        optimizer: OptimizerBase
) -> Tuple[float, float]:
    # Update the model state given a batch of data
    # “Opening” the tape means that TF is going to store the operations done across the context manager. In that way
    # we can come back to those operations and compute the derivatives of those operations in order to compute the
    # gradients required for the backpropagation update.
    with tf.GradientTape() as tape:
        y_pred = model(x=X_batch)
        batch_loss = loss(y_pred=y_pred, y_true=y_batch)
    grads = tape.gradient(target=batch_loss, sources=model.variables)
    optimizer.update_step(gradient=grads, variable=model.variables)
    batch_acc = acc(y_pred=y_pred, y_true=y_batch)
    print("\n")
    return batch_loss, batch_acc, y_pred, grads


@tf.function
def _validation_step(
        X_batch: tf.Tensor,
        y_batch: tf.Tensor,
        loss: Callable,
        acc: Callable,
        model: ModelBase
) -> Tuple[float, float, List[tf.Variable], List[tf.Variable]]:
    # Evaluate the model on given a batch of validation data
    y_pred = model(x=X_batch)
    batch_loss = loss(y_pred=y_pred, y_true=y_batch)
    batch_acc = acc(y_pred=y_pred, y_true=y_batch)
    return batch_loss, batch_acc


def train_model(
        model: ModelBase,
        train_Xy: tf.data.Dataset,
        val_Xy: tf.data.Dataset,
        loss: Callable,
        acc: Callable,
        optimizer: OptimizerBase,
        batch_size: int = -1,
        epochs: int = 500,
        log_flag: bool = True,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    # Initialize data structures
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    train_y_preds, train_grads, train_model_wbs = [], [], []

    """
    Batching the dataset creates a tuple with N elements, where element contains the batch size number of elements of 
    the original dataset.
    
    Hence, if batch_size equal to:
        - -1: Keyword to indicate to do not batch, in other words, create a single batch with all elements of the 
            dataset N.
        - 1: Creates N batches where each one contains a single entry from the original dataset.
        - b: Where 1 < b < N, creates (N // b + 1) batches with b elements inside each batch.
        - N (or s > N): Same effect than -1.
    """
    if batch_size == -1:
        batch_size = len(train_Xy)
    train_Xy = train_Xy.batch(batch_size=batch_size)
    val_Xy = val_Xy.batch(batch_size=batch_size)

    # Format training loop and begin training
    for epoch in range(epochs):
        batch_losses_train, batch_accs_train = [], []
        batch_losses_val, batch_accs_val = [], []

        # Iterate over the training data; X_batch and y_batch have a shape [BS, len(shape_input_array)]
        for step, (X_batch, y_batch) in enumerate(train_Xy):
            # Compute gradients and update the model's parameters
            batch_loss, batch_acc, y_pred, grads = _train_step(
                X_batch=X_batch, y_batch=y_batch, loss=loss, acc=acc, model=model, optimizer=optimizer
            )
            # Keep track of batch-level training performance and model variables and gradient updates
            train_y_preds.append(y_pred)
            train_grads.append(grads)
            batch_losses_train.append(batch_loss)
            train_model_wbs.append(model.variables)
            batch_accs_train.append(batch_acc)

            # Log every 200 batches.
            if log_flag and step % 200 == 0:
                print(f"Training loss for batch {step + 1} / {len(train_Xy)}: {batch_loss:.4f}")

        # Iterate over the validation data
        for step, (X_batch, y_batch) in enumerate(val_Xy):
            batch_loss, batch_acc = _validation_step(X_batch=X_batch, y_batch=y_batch, loss=loss, acc=acc, model=model)
            batch_losses_val.append(batch_loss)
            batch_accs_val.append(batch_acc)

        # Keep track of epoch-level model performance
        train_losses.append(tf.reduce_mean(input_tensor=batch_losses_train))
        train_accs.append(tf.reduce_mean(input_tensor=batch_accs_train))
        val_losses.append(tf.reduce_mean(input_tensor=batch_losses_val))
        val_accs.append(tf.reduce_mean(input_tensor=batch_accs_val))

        if log_flag:
            print(f"Epoch: {epoch}")
            print(f"Training loss: {train_losses[-1]:.3f}, Training accuracy: {train_accs[-1]:.3f}")
            print(f"Validation loss: {val_losses[-1]:.3f}, Validation accuracy: {val_accs[-1]:.3f}")
    return train_losses, train_accs, val_losses, val_accs
