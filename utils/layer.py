from abc import ABC
import tensorflow as tf
from utils.initialization import init_weights_xavier
from typing import Callable


class LayerBase:
    def __init__(self, out_dims: int, weight_init: Callable, activation: Callable):
        self.out_dims = out_dims
        self.weight_init = weight_init
        self.activation = activation
        self.in_dims = None
        self.built = False

    @classmethod
    def __call__(cls, *args, **kwargs): ...

    @classmethod
    def _forward(cls, *args, **kwargs): ...


class DenseLayer(tf.Module, ABC, LayerBase):
    def __init__(self, out_dims: int, weight_init: Callable = init_weights_xavier, activation: Callable = tf.identity):
        LayerBase.__init__(self, out_dims=out_dims, weight_init=weight_init, activation=activation)

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        if x.dtype != tf.float32:
            x = tf.cast(x=x, dtype=tf.float32)
        if not self.built:  # Initialize weights and biases
            self.in_dims = x.shape[1]  # Infer the input dimension based on first call
            self.w = tf.Variable(initial_value=init_weights_xavier(shape=(self.in_dims, self.out_dims)))
            self.b = tf.Variable(initial_value=tf.zeros(shape=(self.out_dims, )))
            self.built = True
        out = self._forward(x=x)
        return out

    def _forward(self, x: tf.Tensor) -> tf.Tensor:
        # Compute forward pass
        z = tf.add(x=tf.matmul(a=x, b=self.w), y=self.b)
        out = self.activation(z)
        return out
