import tensorflow as tf
from abc import ABC
from typing import Optional, List
from utils.layer import LayerBase


class ModelBase:
    def __init__(self, name: Optional[str]):
        self.name = name

    @tf.function
    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        # Execute the model's layers sequentially
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
