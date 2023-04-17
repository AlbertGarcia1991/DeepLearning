from abc import ABC
from typing import Callable

import tensorflow as tf

from utils.model import ModelBase


class ExportModule(tf.Module, ABC):
    def __init__(self, model: ModelBase, class_pred: Callable, out_name: str):
        # Initialize pre and postprocessing functions
        super().__init__()
        self.out_name = out_name
        self.model = model
        self.class_pred = class_pred

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None, None], dtype=tf.uint8)])
    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        # Run the ExportModule for new data points
        y = self.model(x)
        y = self.class_pred(y)
        return y
