from abc import ABC
from typing import Callable, List, NoReturn, Optional

import tensorflow as tf

from utils.layer import LayerBase
from utils.loss import loss_binary_cross_entropy, loss_mse
from utils.metric import metric_accuracy


class ModelBase:
    def __init__(self, name: Optional[str]):
        self.name = name

    @tf.function
    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        """
        Execute the model's layers sequentially.

        Args:
            x: Input of the model.

        Returns:
            x: Output from the last layer of the model.
        """
        for layer in self.layers:
            x = layer(x)
        return x


class ModelMLP(tf.Module, ABC, ModelBase):
    def __init__(self, layers: List[LayerBase], name: Optional[str] = None):
        """
        Initialize ModelMLP with the list of layers. Each item inside the given list contains each layer as a child
        of ModelBase such as DenseLayer, BNLayer, among others.

        Args:
            layers: List of Layers objects to build the ModelMLP model.
        """
        super().__init__(name=name)
        self.layers = layers


class ModelPolynomialRegression:
    def __init__(
            self,
            order: int,
            learning_rate: float = 1e-2,
            loss: Callable = tf.losses.mean_squared_error,
            name: Optional[str] = None,
            hist: bool = True
    ):
        self.order = order
        self.name = name
        self.learning_rate = learning_rate
        self.loss = loss
        self.hist = hist
        self.built = False
        if self.hist:
            self.coefficients_hist = [[] for _ in range(order + 1)]
            self.train_grads_hist = []
            self.train_loss_hist = []

    def reset(self) -> NoReturn:
        self.coefficients = [tf.Variable(0., dtype=tf.float32) for _ in range(self.order + 1)]
        if self.hist:
            self.coefficients_hist = [[] for _ in range(self.order + 1)]
            self.train_grads_hist = []
            self.train_loss_hist = []
        self.built = False

    def fit(self, X_train: tf.Tensor, y_train: tf.Tensor, epochs: int = 50, log_flag: bool = True) -> NoReturn:
        if not self.built:
            self.coefficients = [tf.Variable(0., dtype=tf.float32) for _ in range(order + 1)]
            self.built = True
        X_train = tf.cast(x=X_train, dtype=tf.float32)
        y_train = tf.cast(x=y_train, dtype=tf.float32)
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                y_pred = sum([self.coefficients[i] * (X_train ** i) for i in range(self.order + 1)])
                loss_val = self.loss(y_pred=y_pred, y_true=y_train)
            grads = tape.gradient(target=loss_val, sources=self.coefficients)
            for idx, grad in enumerate(grads):
                self.coefficients[idx].assign_sub(grad * self.learning_rate)
            if self.hist:
                for idx, coef in enumerate(self.coefficients):
                    self.coefficients_hist[idx].append(coef)
                self.train_grads_hist.append(grads)
                self.train_loss_hist.append(loss_val)
            if log_flag:
                print(f"Epoch {epoch + 1} / {epochs}")
                print(f"Training loss: {loss_val:.3f}")

    def __call__(self, X: tf.Tensor) -> tf.Tensor:
        if self.built:
            return sum([self.coefficients[i] * (X ** i) for i in range(self.order + 1)])
        else:
            raise ValueError("Cannot run inference until the model has been fit")


class ModelLogisticRegression:
    def __init__(
            self,
            learning_rate: float = 1e-2,
            loss: Callable = loss_binary_cross_entropy,
            name: Optional[str] = None,
            hist: bool = True
    ):
        self.name = name
        self.learning_rate = learning_rate
        self.loss = loss
        self.hist = hist
        self.built = False
        if self.hist:
            self.train_W_hist = []
            self.train_b_hist = []
            self.train_grads_hist = []
            self.train_loss_hist = []
            self.train_accs = []

    def reset(self) -> NoReturn:
        self.W = None
        self.b = None
        if self.hist:
            self.train_W_hist = []
            self.train_b_hist = []
            self.train_grads_hist = []
            self.train_loss_hist = []
            self.train_accs = []
        self.built = False

    def fit(self, X_train: tf.Tensor, y_train: tf.Tensor, epochs: int = 50, log_flag: bool = True) -> NoReturn:
        if not self.built:
            self.W = tf.Variable(initial_value=tf.zeros(shape=(X_train.shape[-1], 1), dtype=tf.float32))
            self.b = tf.Variable(initial_value=tf.zeros(shape=(1,), dtype=tf.float32))
            self.multinomial_flag = True if y_train.shape[1] > 1 else False
            self.built = True

        X_train = tf.cast(x=X_train, dtype=tf.float32)
        y_train = tf.cast(x=y_train, dtype=tf.float32)
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                logits = self(X=X_train)
                loss_val = self.loss(y_true=y_train, y_pred=logits)
            grads = tape.gradient(target=loss_val, sources=[self.W, self.b])
            self.W.assign_sub(grads[0] * self.learning_rate)
            self.b.assign_sub(grads[1] * self.learning_rate)
            acc = metric_accuracy(y_pred=self(X=X_train), y_true=y_train)
            if self.hist:
                self.train_grads_hist.append(grads)
                self.train_W_hist.append(self.W)
                self.train_b_hist.append(self.b)
                self.train_loss_hist.append(loss_val)
                self.train_accs.append(acc)

            if log_flag:
                print(f"Epoch {epoch + 1} / {epochs}")
                print(f"Training loss: {loss_val:.3f}, Training accuracy: {acc:.2%}")

    def __call__(self, X: tf.Tensor) -> tf.Tensor:
        if self.built:
            logits = tf.matmul(X, self.W) + self.b
            if self.multinomial_flag:
                return tf.nn.softmax(logits)
            else:
                return tf.sigmoid(logits)
        else:
            raise ValueError("Cannot run inference until the model has been fit")

    def predict(self, X: tf.Tensor, threshold: float = 0.5) -> tf.Tensor:
        if self.multinomial_flag:
            return tf.argmax(self(X=X), axis=1)
        else:
            return 1 if self(X=X) > threshold else 0
