import tensorflow as tf
from abc import ABC
from typing import Optional, List, Callable, NoReturn
from utils.layer import LayerBase
from utils.loss import loss_mse


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


class ModelLinearRegression:
    def __init__(
            self,
            learning_rate: float = 1e-2,
            loss: Callable = loss_mse,
            name: Optional[str] = None,
            hist: bool = True
    ):
        self.name = name
        self.learning_rate = learning_rate
        self.loss = loss
        self.hist = hist
        self.a = tf.Variable(0.)
        self.b = tf.Variable(0.)
        self.built = False
        if self.hist:
            self.a_hist = []
            self.b_hist = []
            self.grads_hist = []
            self.loss_hist = []

    def reset(self) -> NoReturn:
        self.a = tf.Variable(0.)
        self.b = tf.Variable(0.)
        if self.hist:
            self.a_hist = []
            self.b_hist = []
            self.grads_hist = []
            self.loss_hist = []
        self.built = False

    def fit(self, x_train: tf.Tensor, y_train: tf.Tensor, epochs: int = 50) -> NoReturn:
        x_train = tf.cast(x=x_train, dtype=tf.float32)
        y_train = tf.cast(x=y_train, dtype=tf.float32)
        for step in range(epochs):
            with tf.GradientTape() as tape:
                y_pred = self.a * x_train + self.b
                loss = self.loss(y_pred=y_pred, y_true=y_train)
            grads = tape.gradient(target=loss, sources=[self.a, self.b])
            self.a.assign_sub(grads[0] * self.learning_rate)
            self.b.assign_sub(grads[1] * self.learning_rate)
            if self.hist:
                self.grads_hist.append(grads)
                self.a_hist.append(self.a)
                self.b_hist.append(self.b)
                self.loss_hist.append(loss)
        self.built = True

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        if self.built:
            return self.a * x + self.b
        else:
            raise ValueError("Cannot run inference until the model has been fit")

