from typing import List

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as sk_metrics
import tensorflow as tf

matplotlib.use('TkAgg')


def plot_training_metrics(
        train_metric: List[float], val_metric: List[float], metric_type: str, return_flag: bool = False
) -> plt.Figure:
    # Visualize metrics vs training Epochs
    plt.figure(figsize=(20, 10))
    plt.plot(range(len(train_metric)), train_metric, label=f"Training {metric_type}")
    plt.plot(range(len(val_metric)), val_metric, label=f"Validation {metric_type}")
    plt.xlabel("Epochs")
    plt.ylabel(metric_type)
    plt.legend()
    plt.title(f"{metric_type} vs Training epochs")
    if return_flag:
        return plt
    plt.show()


def plot_confusion_matrix(
        y_pred: tf.Tensor, y_true: tf.Tensor, n_classes: int, norm: bool = True, return_flag: bool = False
) -> plt.Figure:
    plt.figure(figsize=(n_classes, n_classes))
    confusion = sk_metrics.confusion_matrix(
        tf.argmax(y_pred, axis=1).numpy(),
        tf.argmax(y_true, axis=1).numpy()
    )
    if norm:
        norm_tensor = tf.reduce_sum(confusion, axis=1)
        confusion = tf.transpose(confusion) / norm_tensor
        confusion = tf.transpose(tf.where(tf.math.is_inf(confusion) | tf.math.is_nan(confusion), 0, confusion))
    ax = sns.heatmap(
        confusion, xticklabels=range(n_classes), yticklabels=range(n_classes),
        cmap="Blues", annot=True, fmt=".2%" if norm else ".0f", square=True)
    plt.title("Confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    if return_flag:
        return plt
    plt.show()
