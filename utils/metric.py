from typing import Dict, List, Optional, Union

import tensorflow as tf


def metric_accuracy(y_pred: tf.Tensor, y_true: tf.Tensor) -> float:
    class_preds = tf.argmax(input=y_pred, axis=1)
    is_equal = tf.equal(x=tf.argmax(y_true, 1), y=class_preds)
    acc = tf.reduce_mean(input_tensor=tf.cast(x=is_equal, dtype=tf.float32))
    return acc


def metric_accuracy_per_class(
        n_classes: int,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        print_flag: bool = False
) -> Optional[Dict[Union[str, float], float]]:
    label_acc_dict = {}
    for label in range(n_classes):
        label_idxs = (tf.argmax(y_true, axis=1) == label)
        count_label_true = tf.math.count_nonzero(label_idxs)
        count_label_pred = tf.math.count_nonzero(tf.argmax(y_pred, axis=1)[label_idxs])
        label_acc_dict[label] = count_label_pred / count_label_true

    if print_flag:
        for label, acc in label_acc_dict.items():
            print(f"Digit {label}: {acc:.2%}")

    return label_acc_dict
