import pickle
from abc import ABC
from typing import List, Optional, Tuple

import tensorflow as tf
from numpy import ndarray, unique

from datasets.io_datasets import get_mnist
from utils.layer import BNLayer, DenseLayer
from utils.loss import loss_cross_entropy
from utils.metric import metric_accuracy, metric_accuracy_per_class
from utils.model import ModelMLP
from utils.op_tensor import op_flatten_input, op_one_hot
from utils.optimizer import *
from utils.plot import plot_confusion_matrix, plot_training_metrics
from utils.train import train_model


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
        500,
        200,
        100,
    ]
    output_dims = 10

    # Custom optimizers are not wrapped around Keras optimizers, thus, loading a tf.keras.optimizer object will not work
    opts = [
        OptimizerAdam(),
        OptimizerSGD(),
        OptimizerNadam(),
        OptimizerAdaBound(),
        OptimizerAdaGrad(),
        OptimizerNAG(),
        OptimizerRMSprop(),
        OptimizerSGDMomentum()
    ]

    bns = [False, True]
    bss = [1, 16, 64, 512, 1024, -1]

    for idx1, opt in enumerate(opts):
        for idx2, bn_flag in enumerate(bns):
            for idx3, bs in enumerate(bss):
                current_iter = idx3 + idx2 * len(bns) + idx1 * len(opts)
                print(f"Current iteration: {current_iter} / {len(opts) * len(bns) * len(bss)}")

                # We can either use custom LayerBase objects or directly load tf.keras off-the-shelf layers. Example:
                #   mlp_layers.append(tf.keras.layers.Dense(units=10, activation=tf.keras.activation.softmax))
                mlp_layers = [
                    DenseLayer(out_dims=out_layer_dims, activation=tf.nn.relu) for out_layer_dims in hidden_layers_dims
                ]
                if bn_flag:
                    mlp_layers.append(
                        BNLayer(out_dims=hidden_layers_dims[-1])
                    )
                mlp_layers.append(DenseLayer(out_dims=output_dims, activation=tf.nn.relu, soft_max_flag=True))
                mlp_model = ModelMLP(layers=mlp_layers)

                train_losses, train_accs, val_losses, val_accs, train_model_wbs = train_model(
                    model=mlp_model,
                    train_Xy=train_Xy,
                    val_Xy=test_Xy,
                    loss=loss_cross_entropy,
                    acc=metric_accuracy,
                    optimizer=opt,
                    batch_size=bs,
                    epochs=1000,
                    early_stop=["acc", 1520, 1]
                )

                y_pred = mlp_model(X_test)

                loss_plot = plot_training_metrics(train_losses, val_losses, "Cross-Entropy Loss", return_flag=True)
                acc_plot = plot_training_metrics(train_accs, val_accs, "Accuracy", return_flag=True)
                cm = metric_accuracy_per_class(n_classes=n_classes, y_true=y_test, y_pred=y_pred, print_flag=False)
                cm_plot = plot_confusion_matrix(y_true=y_test, y_pred=y_pred, n_classes=n_classes)

                filename = f"EXP{current_iter}-OPT_{opt.name}-BS_{bs}-BN_{bn_flag}.pkl"
                with open(filename, "wb") as f:
                    obj = {
                        "model": mlp_model,
                        "train_model_wbs": train_model_wbs,
                        "optimizer": opt,
                        "batch_size": bs,
                        "batch_norm_flag": bn_flag,
                        "cm": cm,
                        "plot": {
                            "loss": loss_plot,
                            "acc": acc_plot,
                            "cm_plot": cm_plot
                        }
                    }
                    pickle.dump(obj=obj, file=f)
