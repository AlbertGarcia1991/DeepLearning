from abc import ABC
from typing import Callable, Optional, Tuple

import tensorflow as tf

from utils.initialization import init_weights_xavier


class LayerBase:
    def __init__(
            self,
            out_dims: int,
            weight_init: Optional[Callable] = None,
            activation: Optional[Callable] = None,
            trainable: Optional[bool] = None
    ):
        self.out_dims = out_dims
        self.weight_init = weight_init
        self.activation = activation
        self.trainable = trainable
        self.in_dims = None
        self.built = False

    @classmethod
    def __call__(cls, *args, **kwargs): ...

    @classmethod
    def _forward(cls, *args, **kwargs): ...


class DenseLayer(tf.Module, ABC, LayerBase):
    def __init__(
            self,
            out_dims: int,
            weight_init: Callable = init_weights_xavier,
            activation: Callable = tf.identity,
            trainable: Optional[bool] = None,
            soft_max_flag: bool = False
    ):
        LayerBase.__init__(self, out_dims=out_dims, weight_init=weight_init, activation=activation, trainable=trainable)
        self.soft_max_flag = soft_max_flag

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        if x.dtype != tf.float32:
            x = tf.cast(x=x, dtype=tf.float32)
        if not self.built:  # Initialize weights and biases
            self.in_dims = x.shape[1]  # Infer the input dimension based on first call
            self.w = tf.Variable(
                initial_value=init_weights_xavier(shape=(self.in_dims, self.out_dims)),
                dtype=tf.float32,
                trainable=self.trainable,
                name="a"
            )
            self.b = tf.Variable(
                initial_value=tf.zeros(shape=(self.out_dims,)),
                dtype=tf.float32,
                trainable=self.trainable,
                name="b"
            )
            self.built = True
        out = self._forward(x=x)
        return out

    def _forward(self, x: tf.Tensor) -> tf.Tensor:
        z = tf.add(x=tf.matmul(a=x, b=self.w), y=self.b)
        out = self.activation(z)
        if self.soft_max_flag:
            out = tf.nn.softmax(logits=out)
        return out


class BNLayer(tf.Module, ABC, LayerBase):
    def __init__(self, out_dims: int, eps: float = 1e-5, momentum: float = 0.9, trainable: bool = True):
        LayerBase.__init__(self, out_dims=out_dims, trainable=trainable)
        self.eps = eps
        self.momentum = momentum

    def _true_fn(self) -> Tuple[tf.Tensor, tf.Tensor]:
        # Update moving_mean and moving_variance
        self.moving_mean.assign(value=self.moving_mean * self.momentum + self.batch_mean * (1 - self.momentum))
        self.moving_variance.assign(
            value=self.moving_variance * self.momentum + self.batch_variance * (1 - self.momentum)
        )
        return self.batch_mean, self.batch_variance

    def _false_fn(self) -> Tuple[tf.Tensor, tf.Tensor]:
        return self.moving_mean, self.moving_variance

    def __call__(self, x: tf.Tensor, training: bool = True) -> tf.Tensor:
        if not self.built:
            self.gamma = tf.Variable(
                initial_value=tf.ones([self.out_dims]), dtype=tf.float32, trainable=self.trainable, name="gamma"
            )
            self.beta = tf.Variable(
                initial_value=tf.zeros([self.out_dims]), dtype=tf.float32, trainable=self.trainable, name="beta"
            )
            self.moving_mean = tf.Variable(
                initial_value=tf.zeros([self.out_dims]), dtype=tf.float32, trainable=False, name="moving_mean"
            )
            self.moving_variance = tf.Variable(
                initial_value=tf.ones([self.out_dims]), dtype=tf.float32, trainable=False, name="moving_variance"
            )
            self.built = True
        out = self._forward(x=x, training=training)
        return out

    def _forward(self, x: tf.Tensor, training: bool) -> tf.Tensor:
        if training and self.trainable:
            self.batch_mean, self.batch_variance = tf.nn.moments(x, [0])
            mean, variance = tf.cond(
                pred=tf.equal(x=self.trainable, y=True),
                true_fn=self._true_fn,
                false_fn=self._false_fn
            )
        else:
            mean, variance = self.moving_mean, self.moving_variance
        out = tf.nn.batch_normalization(
            x=x,
            mean=mean,
            variance=variance,
            offset=self.beta,
            scale=self.gamma,
            variance_epsilon=self.eps
        )
        return out
