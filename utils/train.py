from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import tensorflow as tf
from numpy import array, ndarray

from utils.model import ModelBase
from utils.optimizer import OptimizerBase


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
    return batch_loss, batch_acc


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
        early_stop: Optional[Tuple[str, int, float]] = None
) -> Tuple[List[float], List[float], List[float], List[float], ndarray]:
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    train_model_wbs = []

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
    val_Xy = val_Xy.batch(batch_size=len(val_Xy))  # Validation does not require batch processing

    # Format training loop and begin training
    for epoch in range(epochs):
        train_batch_losses, train_batch_accs = [], []
        val_batch_losses, val_batch_accs = [], []
        train_model_wbs_ = None

        # Iterate over the training data; X_batch and y_batch have a shape [BS, len(shape_input_array)]
        for step, (X_batch, y_batch) in enumerate(train_Xy):
            # Compute gradients and update the model's parameters
            batch_loss, batch_acc = _train_step(
                X_batch=X_batch, y_batch=y_batch, loss=loss, acc=acc, model=model, optimizer=optimizer
            )
            # Keep track of batch-level training performance and model variables
            train_batch_losses.append(batch_loss)
            train_batch_accs.append(batch_acc)
            if train_model_wbs_ is None:
                train_model_wbs_ = array(
                    [model.trainable_variables[i].numpy() for i in range(len(model.trainable_variables))], dtype=object)
            else:
                train_model_wbs_ += array(
                    [model.trainable_variables[i].numpy() for i in range(len(model.trainable_variables))], dtype=object)

            if log_flag and step % 200 == 0:  # Log every 200 batches.
                print(f"Training loss for batch {step + 1} / {len(train_Xy)}: {batch_loss:.4f}")

        train_model_wbs.append(train_model_wbs_ / (step + 1))

        # Iterate over the validation data
        for step, (X_batch, y_batch) in enumerate(val_Xy):
            batch_loss, batch_acc = _validation_step(X_batch=X_batch, y_batch=y_batch, loss=loss, acc=acc, model=model)
            val_batch_losses.append(batch_loss)
            val_batch_accs.append(batch_acc)

        # Keep track of epoch-level model performance
        train_losses.append(tf.reduce_mean(input_tensor=train_batch_losses))
        train_accs.append(tf.reduce_mean(input_tensor=train_batch_accs))
        val_losses.append(tf.reduce_mean(input_tensor=val_batch_losses))
        val_accs.append(tf.reduce_mean(input_tensor=val_batch_accs))

        # Early stop implementation
        if early_stop is not None:
            computed_delta = None
            metric = early_stop[0].lower()
            patience = early_stop[1]
            min_delta = early_stop[2] / 100
            early_stop_flag = False
            if len(val_losses) >= patience:
                if metric == "accuracy" or metric == "acc":
                    computed_delta = (val_accs[-1] - val_accs[-patience]) / val_accs[-patience]
                if metric == "loss":
                    computed_delta = -(val_losses[-1] - val_losses[-patience]) / val_losses[-patience]
                if computed_delta is not None and computed_delta < min_delta:
                    early_stop_flag = True
            if early_stop_flag:
                print(f"Early stop after patience of {patience} epochs checking metric: {metric}")
                break

        if log_flag:
            print(f"Epoch {epoch + 1} / {epochs}")
            print(f"Training loss: {train_losses[-1]:.3f}, Training accuracy: {train_accs[-1]:.3f}")
            print(f"Validation loss: {val_losses[-1]:.3f}, Validation accuracy: {val_accs[-1]:.3f}\n\n")

    return train_losses, train_accs, val_losses, val_accs, array(train_model_wbs)
