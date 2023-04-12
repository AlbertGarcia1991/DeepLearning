import tensorflow as tf
from typing import Optional


class ModelBase:
    def __init__(self, name: Optional[str]):
        self.name = name

    @tf.function
    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        # Execute the model's layers sequentially
        for layer in self.layers:
            x = layer(x)
        return x
