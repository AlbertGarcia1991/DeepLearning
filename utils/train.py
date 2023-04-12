import tensorflow as tf
from utils.model import ModelBase
from utils.optimizer import OptimizerBase
from typing import Callable, Tuple, List


@tf.function
def _train_step(
        x_batch: tf.Tensor,
        y_batch: tf.Tensor,
        loss: Callable,
        acc: Callable,
        model: ModelBase,
        optimizer: OptimizerBase
) -> Tuple[float, float]:
    # Update the model state given a batch of data
    # TODO: definition of GradientTape() and subsequent methods
    with tf.GradientTape() as tape:
        y_pred = model(x=x_batch)
        batch_loss = loss(y_pred=y_pred, y_true=y_batch)
    grads = tape.gradient(target=batch_loss, sources=model.variables)
    optimizer.apply_gradients(grads=grads, variables=model.variables)
    batch_acc = acc(y_pred=y_pred, y_true=y_batch)
    return batch_loss, batch_acc


@tf.function
def _validation_step(
        x_batch: tf.Tensor,
        y_batch: tf.Tensor,
        loss: Callable,
        acc: Callable,
        model: ModelBase
) -> Tuple[float, float]:
    # Evaluate the model on given a batch of validation data
    y_pred = model(x=x_batch)
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

    if batch_size == -1:
        batch_size = len(train_Xy)
    train_Xy = train_Xy.batch(batch_size=batch_size)
    val_Xy = val_Xy.batch(batch_size=batch_size)

    # Format training loop and begin training
    for epoch in range(epochs):
        batch_losses_train, batch_accs_train = [], []
        batch_losses_val, batch_accs_val = [], []

        # Iterate over the training data
        for step, (x_batch, y_batch) in enumerate(train_Xy):
            # Compute gradients and update the model's parameters
            batch_loss, batch_acc = _train_step(x_batch=x_batch, y_batch=y_batch, loss=loss, acc=acc, model=model,
                                                optimizer=optimizer)
            # Keep track of batch-level training performance
            batch_losses_train.append(batch_loss)
            batch_accs_train.append(batch_acc)

            # Log every 200 batches.
            if log_flag and step % 200 == 0:
                print(f"Training loss (for one batch) at step {step} / {len(train_Xy)}: {batch_loss:.4f}")

        # Iterate over the validation data
        for step, (x_batch, y_batch) in enumerate(val_Xy):
            batch_loss, batch_acc = _validation_step(x_batch=x_batch, y_batch=y_batch, loss=loss, acc=acc, model=model)
            batch_losses_val.append(batch_loss)
            batch_accs_val.append(batch_acc)

            if log_flag and step % 200 == 0:
                print(f"Test loss (for one batch) at step {step} / {len(val_Xy)}: {batch_loss:.4f}")

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
