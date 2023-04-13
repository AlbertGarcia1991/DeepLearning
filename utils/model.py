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
        self.coefficients = [tf.Variable(0., dtype=tf.float32) for _ in range(order + 1)]
        self.built = False
        if self.hist:
            self.coefficients_hist = [[] for _ in range(order + 1)]
            self.grads_hist = []
            self.loss_hist = []

    def reset(self) -> NoReturn:
        self.coefficients = [tf.Variable(0., dtype=tf.float32) for _ in range(self.order + 1)]
        if self.hist:
            self.coefficients_hist = [[] for _ in range(self.order + 1)]
            self.grads_hist = []
            self.loss_hist = []
        self.built = False

    def fit(self, x_train: tf.Tensor, y_train: tf.Tensor, epochs: int = 50) -> NoReturn:
        x_train = tf.cast(x=x_train, dtype=tf.float32)
        y_train = tf.cast(x=y_train, dtype=tf.float32)
        for step in range(epochs):
            with tf.GradientTape() as tape:
                y_pred = sum([self.coefficients[i] * (x_train ** i) for i in range(self.order + 1)])
                loss_val = self.loss(y_pred=y_pred, y_true=y_train)
            grads = tape.gradient(target=loss_val, sources=self.coefficients)
            for idx, grad in enumerate(grads):
                self.coefficients[idx].assign_sub(grad * self.learning_rate)
            if self.hist:
                for idx, coef in enumerate(self.coefficients):
                    self.coefficients_hist[idx].append(coef)
                self.grads_hist.append(grads)
                self.loss_hist.append(loss_val)
        self.built = True

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        if self.built:
            return sum([self.coefficients[i] * (x ** i) for i in range(self.order + 1)])
        else:
            raise ValueError("Cannot run inference until the model has been fit")

