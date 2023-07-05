from typing import Tuple

import tensorflow as tf
from numpy import ndarray, unique

from datasets.io_datasets import get_mnist
from utils.model import ModelLogisticRegression
from utils.op_tensor import op_flatten_input, op_one_hot


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

    clf = ModelLogisticRegression(learning_rate=0.5)
    clf.fit(X_train=X_train, y_train=y_train)
