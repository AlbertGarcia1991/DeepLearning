import numpy as np
from abc import ABC
import matplotlib.pyplot as plt
from datasets.io_datasets import get_mnist
import tensorflow as tf
from numpy import ndarray
from typing import List, Tuple, Callable, Optional
import tempfile
import os
import sklearn.metrics as sk_metrics
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')


def plot_training_metrics(train_metric: List[float], val_metric: List[float], metric_type: str):
    # Visualize metrics vs training Epochs
    plt.figure(figsize=(20, 10))
    plt.plot(range(len(train_metric)), train_metric, label=f"Training {metric_type}")
    plt.plot(range(len(val_metric)), val_metric, label=f"Validation {metric_type}")
    plt.xlabel("Epochs")
    plt.ylabel(metric_type)
    plt.legend()
    plt.title(f"{metric_type} vs Training epochs")
    plt.show()


def op_flatten_input(X: ndarray) -> tf.Tensor:
    out_dims = tf.reduce_prod(X.shape[1:]).numpy()
    return tf.reshape(X, (-1, out_dims))


def op_one_hot(y: np.ndarray, depth: Optional[int] = None) -> tf.Tensor:
    if depth is None:
        depth = len(np.unique(y))
    return tf.one_hot(indices=y, depth=depth)


def weights_init_xavier(shape: Tuple[int, int]) -> tf.Tensor:
    in_dims, out_dims = shape
    xavier_lim = tf.sqrt(x=6.) / tf.sqrt(x=tf.cast(x=in_dims + out_dims, dtype=tf.float32))
    weight_vals = tf.random.uniform(shape=shape, minval=-xavier_lim, maxval=xavier_lim, seed=13)
    return weight_vals


def loss_cross_entropy(y_pred: tf.Tensor, y_true: tf.Tensor) -> float:
    # Compute cross-entropy loss with a sparse operation
    sparse_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_true, 1), logits=y_pred)
    return tf.reduce_mean(input_tensor=sparse_ce)


def metric_accuracy(y_pred: tf.Tensor, y_true: tf.Tensor) -> float:
    # Compute accuracy after extracting class predictions
    class_preds = tf.argmax(input=tf.nn.softmax(logits=y_pred), axis=1)
    is_equal = tf.equal(x=tf.argmax(y_true, 1), y=class_preds)
    return tf.reduce_mean(input_tensor=tf.cast(x=is_equal, dtype=tf.float32))


class OptimizerAdam:
    def __init__(self, learning_rate: float = 1e-3, beta_1: float = 0.9, beta_2: float = 0.999, eps: float = 1e-7):
        # Initialize optimizer parameters and variable slots
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.learning_rate = learning_rate
        self.eps = eps
        self.t = 1.
        self.v_dvar, self.s_dvar = [], []
        self.built = False

    def apply_gradients(self, grads: tf.Tensor, variables: List[tf.Variable]):
        # Initialize variables on the first call
        if not self.built:
            for var in variables:
                v = tf.Variable(tf.zeros(shape=var.shape))
                s = tf.Variable(tf.zeros(shape=var.shape))
                self.v_dvar.append(v)
                self.s_dvar.append(s)
            self.built = True
        # Update the model variables given their gradients
        for i, (d_var, var) in enumerate(zip(grads, variables)):
            self.v_dvar[i].assign(self.beta_1 * self.v_dvar[i] + (1 - self.beta_1) * d_var)
            self.s_dvar[i].assign(self.beta_2 * self.s_dvar[i] + (1 - self.beta_2) * tf.square(x=d_var))
            v_dvar_bc = self.v_dvar[i] / (1 - (self.beta_1**self.t))
            s_dvar_bc = self.s_dvar[i] / (1 - (self.beta_2**self.t))
            var.assign_sub(delta=self.learning_rate * (v_dvar_bc / (tf.sqrt(x=s_dvar_bc) + self.eps)))
        self.t += 1.
        return


class DenseLayer(tf.Module, ABC):
    def __init__(self, out_dims: int, weight_init: Callable = weights_init_xavier, activation: Callable = tf.identity):
        super().__init__()
        self.out_dims = out_dims
        self.weight_init = weight_init
        self.activation = activation
        self.built = False

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        x = tf.cast(x=x, dtype=tf.float32)
        if not self.built:
            # Infer the input dimension based on first call
            self.in_dims = x.shape[1]

            # Initialize weights and biases
            self.w = tf.Variable(initial_value=weights_init_xavier(shape=(self.in_dims, self.out_dims)))
            self.b = tf.Variable(initial_value=tf.zeros(shape=(self.out_dims, )))
            self.built = True

        # Compute forward pass
        z = tf.add(x=tf.matmul(a=x, b=self.w), y=self.b)
        return self.activation(z)


class ModelMLP(tf.Module, ABC):
    def __init__(self, layers: List[DenseLayer]):
        """
        Initialize ModelMLP with the list of layers. Each item inside the given list contains each layer as a DenseLayer
        object.

        Args:
            layers: List of DenseLayers to build the ModelMLP model.
        """
        super().__init__()
        self.layers = layers

    @tf.function
    def __call__(self, x: tf.Tensor, preds: bool = False) -> tf.Tensor:
        # Execute the model's layers sequentially
        for layer in self.layers:
            x = layer(x)
        return x


@tf.function
def train_step(
        x_batch: tf.Tensor,
        y_batch: tf.Tensor,
        loss: Callable,
        acc: Callable,
        model: ModelMLP,
        optimizer: OptimizerAdam
) -> Tuple[float, float]:
    # Update the model state given a batch of data
    with tf.GradientTape() as tape:
        y_pred = model(x=x_batch)
        batch_loss = loss(y_pred=y_pred, y_true=y_batch)
    grads = tape.gradient(target=batch_loss, sources=model.variables)
    optimizer.apply_gradients(grads=grads, variables=model.variables)
    batch_acc = acc(y_pred=y_pred, y_true=y_batch)
    return batch_loss, batch_acc


@tf.function
def validation_step(x_batch: tf.Tensor, y_batch: tf.Tensor, loss: Callable, acc: Callable, model: ModelMLP
                    ) -> Tuple[float, float]:
    # Evaluate the model on given a batch of validation data
    y_pred = model(x=x_batch)
    batch_loss = loss(y_pred=y_pred, y_true=y_batch)
    batch_acc = acc(y_pred=y_pred, y_true=y_batch)
    return batch_loss, batch_acc


def train_model(
        model: ModelMLP,
        train_Xy: tf.data.Dataset,
        val_Xy: tf.data.Dataset,
        loss: Callable,
        acc: Callable,
        optimizer: OptimizerAdam,
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
            batch_loss, batch_acc = train_step(
                x_batch=x_batch, y_batch=y_batch, loss=loss, acc=acc, model=model, optimizer=optimizer
            )
            # Keep track of batch-level training performance
            batch_losses_train.append(batch_loss)
            batch_accs_train.append(batch_acc)

            # Log every 200 batches.
            if log_flag and step % 200 == 0:
                print(f"Training loss (for one batch) at step {step} / {len(train_Xy)}: {batch_loss:.4f}")

        # Iterate over the validation data
        for step, (x_batch, y_batch) in enumerate(val_Xy):
            batch_loss, batch_acc = validation_step(x_batch=x_batch, y_batch=y_batch, loss=loss, acc=acc, model=model)
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


class ExportModule(tf.Module, ABC):
    def __init__(self, model: ModelMLP, class_pred: Callable, out_name: str):
        super().__init__()
        # Initialize pre and postprocessing functions
        self.out_name = out_name
        self.model = model
        self.class_pred = class_pred

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None, None], dtype=tf.uint8)])
    def __call__(self, x: tf.Tensor):
        # Run the ExportModule for new data points
        y = self.model(x)
        y = self.class_pred(y)
        return y


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = get_mnist()

    if len(np.unique(y_train)) != len(np.unique(y_test)):
        raise ValueError("Not all classes are present in either train or test y_true array")
    else:
        n_classes = len(np.unique(y_train))

    def preprocess(X_train: ndarray, y_train: ndarray, X_test: ndarray, y_test: ndarray
                   ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        X_train = op_flatten_input(X=X_train)
        X_test = op_flatten_input(X=X_test)
        y_train = op_one_hot(y=y_train, depth=n_classes)
        y_test = op_one_hot(y=y_test, depth=n_classes)

        X_train = X_train / 255
        X_test = X_test / 255

        return X_train, y_train, X_test, y_test


    def class_pred_test(y: tf.Tensor) -> tf.Tensor:
        # Generate class predictions from MLP output
        return tf.argmax(tf.nn.softmax(y), axis=1)

    X_train, y_train, X_test, y_test = preprocess(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    train_Xy = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_Xy = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    hidden_layers_dims = [
        700,
        500,
    ]
    output_dims = 10

    mlp_layers = [DenseLayer(out_dims=out_layer_dims, activation=tf.nn.relu) for out_layer_dims in hidden_layers_dims]
    mlp_layers.append(DenseLayer(out_dims=output_dims, activation=tf.nn.relu))
    mlp_model = ModelMLP(layers=mlp_layers)

    optimizer = OptimizerAdam()

    train_losses, train_accs, val_losses, val_accs = train_model(
        model=mlp_model,
        train_Xy=train_Xy,
        val_Xy=test_Xy,
        loss=loss_cross_entropy,
        acc=metric_accuracy,
        optimizer=optimizer,
        batch_size=-1,
        epochs=2
    )

    plot_training_metrics(train_losses, val_losses, "Cross-Entropy Loss")
    plot_training_metrics(train_accs, val_accs, "Accuracy")

    mlp_model_export = ExportModule(model=mlp_model,
                                    class_pred=class_pred_test,
                                    out_name="test")
    # models = tempfile.mkdtemp()
    # save_path = os.path.join(models, os.path.join("models", mlp_model_export.name))
    # tf.saved_model.save(mlp_model_export, mlp_model_export.name)
    #
    # mlp_loaded = tf.saved_model.load(save_path)

    # TODO: As function
    print("Accuracy breakdown by digit:")
    print("---------------------------")
    label_accs = {}
    for label in range(n_classes):
        label_idxs = (tf.argmax(y_test, axis=1) == label)
        count_label_true = tf.math.count_nonzero(label_idxs)
        y_pred = tf.argmax(mlp_model(X_test[label_idxs]), axis=1)
        count_label_pred = tf.math.count_nonzero(y_pred == label)
        label_accs[label] = count_label_pred / count_label_true
    for label, acc in label_accs.items():
        print(f"Digit {label}: {acc:.2%}")


    def show_confusion_matrix(y_pred: tf.Tensor, y_true: tf.Tensor, n_classes: int):
        # TODO: Normalize
        plt.figure(figsize=(n_classes, n_classes))
        confusion = sk_metrics.confusion_matrix(y_pred.numpy(), y_true.numpy())
        ax = sns.heatmap(
            confusion, xticklabels=range(n_classes), yticklabels=range(n_classes),
            cmap="Blues", annot=True, fmt=".0f", square=True)
        plt.title("Confusion matrix")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.show()

    y_test_pred = tf.argmax(mlp_model(X_test), axis=1)
    y_test_true = tf.argmax(y_test, axis=1)

    show_confusion_matrix(y_true=y_test_true, y_pred=y_test_pred, n_classes=n_classes)

    # TODO: Split functions into modules

    # TODO: Hidden layers inspection tools
