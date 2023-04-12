import tensorflow as tf
from typing import List, NoReturn


class OptimizerBase:
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate
        self.built = False

    @classmethod
    def apply_gradients(cls, *args, **kwargs): ...


class OptimizerAdam(OptimizerBase):
    def __init__(self, learning_rate: float = 1e-3, beta_1: float = 0.9, beta_2: float = 0.999, eps: float = 1e-7):
        """
        Initialize optimizer parameters and variable slots.

        Args:
            learning_rate:
            beta_1:
            beta_2:
            eps:
        """
        super().__init__(learning_rate=learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.t = 1.
        self.v_dvar, self.s_dvar = [], []

    def apply_gradients(self, grads: List[tf.Tensor], variables: List[tf.Variable]) -> NoReturn:
        # Initialize variables on the first call
        if not self.built:
            for var in variables:
                v = tf.Variable(initial_value=tf.zeros(shape=var.shape))
                s = tf.Variable(initial_value=tf.zeros(shape=var.shape))
                self.v_dvar.append(v)
                self.s_dvar.append(s)
            self.built = True
        # Update the model variables given their gradients
        for i, (d_var, var) in enumerate(zip(grads, variables)):
            self.v_dvar[i].assign(self.beta_1 * self.v_dvar[i] + (1 - self.beta_1) * d_var)
            self.s_dvar[i].assign(self.beta_2 * self.s_dvar[i] + (1 - self.beta_2) * tf.square(x=d_var))
            v_dvar_bc = self.v_dvar[i] / (1 - (self.beta_1 ** self.t))
            s_dvar_bc = self.s_dvar[i] / (1 - (self.beta_2 ** self.t))
            delta = self.learning_rate * (v_dvar_bc / (tf.sqrt(x=s_dvar_bc) + self.eps))
            var.assign_sub(delta=delta)  # Computes var = var - delta
        self.t += 1.
