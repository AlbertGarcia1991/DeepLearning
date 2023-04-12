from abc import ABC
from datasets.io_datasets import get_mnist
import tensorflow as tf
from numpy import ndarray, unique
from typing import List, Optional, Tuple
from utils.layer import DenseLayer, BNLayer
from utils.op_tensor import op_flatten_input, op_one_hot
from utils.plot import plot_training_metrics, plot_confusion_matrix
from utils.model import ModelMLP
from utils.optimizer import OptimizerAdam
from utils.train import train_model
from utils.metric import metric_accuracy_per_class, metric_accuracy
from utils.loss import loss_cross_entropy


def preprocess(X_train: ndarray, y_train: ndarray, X_test: ndarray, y_test: ndarray
               ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    X_train = op_flatten_input(X=X_train)
    X_test = op_flatten_input(X=X_test)
    y_train = op_one_hot(y=y_train, n_classes=n_classes)
    y_test = op_one_hot(y=y_test, n_classes=n_classes)

    X_train = X_train / 255
    X_test = X_test / 255

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = get_mnist()

    if len(unique(y_train)) != len(unique(y_test)):
        raise ValueError("Not all classes are present in either train or test y_true array")
    else:
        n_classes = len(unique(y_train))

    X_train, y_train, X_test, y_test = preprocess(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    train_Xy = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_Xy = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    hidden_layers_dims = [
        700,
        500,
    ]
    output_dims = 10

    mlp_layers = [DenseLayer(out_dims=out_layer_dims, activation=tf.nn.relu) for out_layer_dims in hidden_layers_dims]
    mlp_layers.append(BNLayer(out_dims=hidden_layers_dims[-1]))
    mlp_layers.append(DenseLayer(out_dims=output_dims, activation=tf.nn.relu, soft_max_flag=True))
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
        epochs=10
    )

    y_pred = mlp_model(X_test)

    plot_training_metrics(train_losses, val_losses, "Cross-Entropy Loss")
    plot_training_metrics(train_accs, val_accs, "Accuracy")
    metric_accuracy_per_class(n_classes=n_classes, y_true=y_test, y_pred=y_pred, print_flag=True)
    plot_confusion_matrix(y_true=y_test, y_pred=y_pred, n_classes=n_classes)

    # TODO: Hidden layers inspection tools
