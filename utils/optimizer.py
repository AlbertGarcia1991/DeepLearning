import tensorflow as tf
from typing import List, NoReturn

"""
Optimizer algorithms overview and selection rule of thumb:

Gradient Descent (GD)
    - Pros: Simple and easy to implement.
    - Cons: Can get stuck in local minima; slow convergence; sensitive to learning rate.
    - When to use: For simple models and convex problems.

Momentum
    - Pros: Accelerates convergence; reduces oscillations.
    - Cons: Can overshoot local minima; sensitive to learning rate and momentum.
    - When to use: When the cost function has a flat or elongated landscape.

Nesterov Accelerated Gradient (NAG)
    - Pros: Faster convergence than Momentum; reduces oscillations.
    - Cons: Can overshoot local minima; sensitive to learning rate and momentum.
    - When to use: Similar to Momentum but with better convergence.

Adagrad
    - Pros: Adapts learning rate per parameter; suitable for sparse data.
    - Cons: Learning rate can become too small and stop model from learning; sensitive to initial learning rate.
    - When to use: For sparse data and convex problems.

RMSprop
    - Pros: Adapts learning rate per parameter; less aggressive update than Adagrad.
    - Cons: Sensitive to initial learning rate and decay rate.
    - When to use: For non-convex problems and when Adagrad learning rate decays too fast.

Adam
    - Pros: Combines Momentum and RMSprop; adaptive learning rates for parameters.
    - Cons: Can show poor convergence for some problems; sensitive to initial learning rate and hyperparameters.
    - When to use: General-purpose optimizer, works well for most deep learning tasks.

Nadam
    - Pros: Combines Nesterov momentum and Adam; faster convergence than Adam.
    - Cons: Sensitive to initial learning rate and hyperparameters.
    - When to use: When you need faster convergence than Adam.

AdaBound
    - Pros: Combines Adam with dynamic learning rate bounds.
    - Cons: Sensitive to initial learning rate and hyperparameters.
    - When to use: When you want to balance adaptive learning rates with bounds.

Lookahead
    - Pros: Enhances other base optimizers; improves convergence and stability.
    - Cons: Introduces additional hyperparameters; increased computation time.
    - When to use: To improve the performance of other optimizers.
"""


class OptimizerBase:

    EPSILON = 1e-7

    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate
        self.built = False

    @classmethod
    def update_step(cls, *args, **kwargs): ...


class OptimizerSGD(OptimizerBase):
    """
    A simple and widely used optimization algorithm that updates model weights using gradients calculated from a
    randomly selected subset (mini-batch) of the training data.
    """
    def __init__(self, learning_rate: float = 1e-3):
        super().__init__(learning_rate=learning_rate)

    def update_step(self, gradient: List[tf.Tensor], variable: List[tf.Variable]) -> NoReturn:
        for d_var, var in zip(gradient, variable):
            if d_var is not None:
                var.assign_sub(self.learning_rate * d_var)


class OptimizerSGDMomentum(OptimizerBase):
    """
    An extension of the basic SGD algorithm that includes a momentum term, which helps the optimizer to accelerate in
    the right direction and reduces oscillations in the learning process.
    """
    def __init__(self, learning_rate: float = 1e-2, momentum: float = 0.9):
        super().__init__(learning_rate=learning_rate)
        self.momentum = momentum
        self.velocity = []

    def update_step(self, gradient: List[tf.Tensor], variable: List[tf.Variable]) -> NoReturn:
        if not self.built:
            for var in variable:
                v = tf.Variable(initial_value=tf.zeros(shape=var.shape))
                self.velocity.append(v)
            self.built = True

        for d_var, var, v in zip(gradient, variable, self.velocity):
            if d_var is not None:
                v.assign(self.momentum * v + self.learning_rate * d_var)
                var.assign_sub(v)


class OptimizerNAG(OptimizerBase):
    """
    A modification of SGD with momentum that computes the gradient at a future position rather than the current
    position, providing better convergence properties.
    """
    def __init__(self, learning_rate: float = 1e-2, momentum: float = 0.9):
        super().__init__(learning_rate=learning_rate)
        self.momentum = momentum
        self.velocity = []

    def update_step(self, gradient: List[tf.Tensor], variable: List[tf.Variable]) -> NoReturn:
        if not self.built:
            for var in variable:
                v = tf.Variable(initial_value=tf.zeros(shape=var.shape))
                self.velocity.append(v)
            self.built = True

        for d_var, var, v in zip(gradient, variable, self.velocity):
            if d_var is not None:
                v.assign(self.momentum * v + self.learning_rate * d_var)
                var.assign_sub(self.momentum * v + self.learning_rate * d_var)


class OptimizerAdaGrad(OptimizerBase):
    """
    An adaptive learning rate method that adjusts the learning rate for each parameter based on the sum of the squared
    gradients, helping with sparse data and learning features with different scales.
    """
    def __init__(self, learning_rate: float = 1e-2):
        super().__init__(learning_rate=learning_rate)
        self.acc_gradient = []

    def update_step(self, gradient: List[tf.Tensor], variable: List[tf.Variable]) -> NoReturn:
        if not self.built:
            for var in variable:
                ag = tf.Variable(initial_value=tf.zeros(shape=var.shape))
                self.acc_gradient.append(ag)
            self.built = True

        for d_var, var, ag in zip(gradient, variable, self.acc_gradient):
            if d_var is not None:
                ag.assign_add(tf.square(d_var))
                var.assign_sub(self.learning_rate * d_var / (tf.sqrt(ag) + self.EPSILON))


class OptimizerAdam(OptimizerBase):
    """
    A widely used optimization algorithm that combines the ideas of RMSprop and momentum, maintaining separate moving
    averages of the gradients and squared gradients, and adapting the learning rate for each parameter.
    """
    def __init__(self, learning_rate: float = 1e-3, beta_1: float = 0.9, beta_2: float = 0.999):
        super().__init__(learning_rate=learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.t = 1.
        self.v_dvar, self.s_dvar = [], []

    def update_step(self, gradient: List[tf.Tensor], variable: List[tf.Variable]) -> NoReturn:
        if not self.built:
            for var in variable:
                v = tf.Variable(initial_value=tf.zeros(shape=var.shape))
                s = tf.Variable(initial_value=tf.zeros(shape=var.shape))
                self.v_dvar.append(v)
                self.s_dvar.append(s)
            self.built = True
        for i, (d_var, var) in enumerate(zip(gradient, variable)):
            if d_var is not None:
                self.v_dvar[i].assign(self.beta_1 * self.v_dvar[i] + (1 - self.beta_1) * d_var)
                self.s_dvar[i].assign(self.beta_2 * self.s_dvar[i] + (1 - self.beta_2) * tf.square(x=d_var))
                v_dvar_bc = self.v_dvar[i] / (1 - (self.beta_1 ** self.t))
                s_dvar_bc = self.s_dvar[i] / (1 - (self.beta_2 ** self.t))
                delta = self.learning_rate * (v_dvar_bc / (tf.sqrt(x=s_dvar_bc) + self.EPSILON))
                var.assign_sub(delta=delta)  # Computes var = var - delta
        self.t += 1.


class OptimizerRMSprop(OptimizerBase):
    """
    A modification of AdaGrad that maintains a moving average of the squared gradients, allowing for better performance
    on non-convex optimization problems.
    """
    def __init__(self, learning_rate: float = 1e-3, decay_rate: float = 0.9):
        super().__init__(learning_rate=learning_rate)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.s_dvar = []

    def update_step(self, gradient: List[tf.Tensor], variable: List[tf.Variable]) -> NoReturn:
        if not self.built:
            for var in variable:
                s = tf.Variable(initial_value=tf.zeros(shape=var.shape))
                self.s_dvar.append(s)
            self.built = True
        for i, (d_var, var) in enumerate(zip(gradient, variable)):
            if d_var is not None:
                self.s_dvar[i].assign(self.decay_rate * self.s_dvar[i] + (1 - self.decay_rate) * tf.square(d_var))
                delta = self.learning_rate * d_var / (tf.sqrt(self.s_dvar[i]) + self.EPSILON)
                var.assign_sub(delta=delta)


class OptimizerNadam(OptimizerBase):
    """
    A combination of Nesterov accelerated gradients and the Adam optimization algorithm, providing the benefits of both
    techniques.
    """
    def __init__(self, learning_rate: float = 1e-3, beta_1: float = 0.9, beta_2: float = 0.999):
        super().__init__(learning_rate=learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.t = 1.
        self.m_dvar, self.v_dvar = [], []

    def update_step(self, gradient: List[tf.Tensor], variable: List[tf.Variable]) -> NoReturn:
        if not self.built:
            for var in variable:
                m = tf.Variable(initial_value=tf.zeros(shape=var.shape))
                v = tf.Variable(initial_value=tf.zeros(shape=var.shape))
                self.m_dvar.append(m)
                self.v_dvar.append(v)
            self.built = True

        for i, (d_var, var) in enumerate(zip(gradient, variable)):
            if d_var is not None:
                self.m_dvar[i].assign(self.beta_1 * self.m_dvar[i] + (1 - self.beta_1) * d_var)
                self.v_dvar[i].assign(self.beta_2 * self.v_dvar[i] + (1 - self.beta_2) * tf.square(d_var))
                m_dvar_bc = self.m_dvar[i] / (1 - self.beta_1 ** self.t)
                v_dvar_bc = self.v_dvar[i] / (1 - self.beta_2 ** self.t)
                delta = self.learning_rate * (
                        self.beta_1 * m_dvar_bc + (1 - self.beta_1) * d_var / (1 - self.beta_1 ** self.t)) / (
                            tf.sqrt(v_dvar_bc) + self.EPSILON
                )
                var.assign_sub(delta=delta)
        self.t += 1.


class OptimizerAdaBound(OptimizerBase):
    """
    A modification of the Adam algorithm that uses dynamic bounds on the learning rates, enabling a smoother transition
    from adaptive learning rates to SGD.
    """
    def __init__(
            self,
            learning_rate: float = 1e-3,
            beta_1: float = 0.9,
            beta_2: float = 0.999,
            final_lr: float = 0.1,
            gamma: float = 1e-3
    ):
        super().__init__(learning_rate=learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.final_lr = final_lr
        self.gamma = gamma
        self.t = 1.
        self.m_dvar, self.v_dvar = [], []

    def update_step(self, gradient: List[tf.Tensor], variable: List[tf.Variable]) -> NoReturn:
        if not self.built:
            for var in variable:
                m = tf.Variable(initial_value=tf.zeros(shape=var.shape))
                v = tf.Variable(initial_value=tf.zeros(shape=var.shape))
                self.m_dvar.append(m)
                self.v_dvar.append(v)
            self.built = True

        for i, (d_var, var) in enumerate(zip(gradient, variable)):
            if d_var is not None:
                self.m_dvar[i].assign(self.beta_1 * self.m_dvar[i] + (1 - self.beta_1) * d_var)
                self.v_dvar[i].assign(self.beta_2 * self.v_dvar[i] + (1 - self.beta_2) * tf.square(d_var))
                m_dvar_bc = self.m_dvar[i] / (1 - self.beta_1 ** self.t)
                v_dvar_bc = self.v_dvar[i] / (1 - self.beta_2 ** self.t)

                lower_bound = self.learning_rate * (1 - 1 / (self.gamma * self.t + 1))
                upper_bound = self.learning_rate * (1 + 1 / (self.gamma * self.t))
                adapted_lr = self.final_lr * m_dvar_bc / (tf.sqrt(v_dvar_bc) + self.EPSILON)
                clipped_lr = tf.clip_by_value(adapted_lr, lower_bound, upper_bound)
                delta = clipped_lr

                var.assign_sub(delta=delta)
        self.t += 1.


class OptimizerLookahead(OptimizerBase):
    """
    An optimization algorithm that works as a wrapper around other optimizers (e.g., SGD, Adam) and updates the
    weights by looking ahead at the sequence of fast weights generated by the inner optimizer.
    """
    def __init__(self, base_optimizer: OptimizerBase, k: int = 5, alpha: float = 0.5):
        super().__init__(learning_rate=base_optimizer.learning_rate)
        self.base_optimizer = base_optimizer
        self.k = k  # Specifies the number of inner optimizer steps during initialization
        self.alpha = alpha  # Controls the interpolation factor
        self.t = 0
        self.backup_vars = []

    def update_step(self, gradient: List[tf.Tensor], variable: List[tf.Variable]) -> NoReturn:
        if not self.built:
            for var in variable:
                backup_var = tf.Variable(initial_value=var)
                self.backup_vars.append(backup_var)
            self.built = True

        self.base_optimizer.update_step(gradient, variable)

        self.t += 1
        if self.t % self.k == 0:
            for i, (var, backup_var) in enumerate(zip(variable, self.backup_vars)):
                new_backup_var = var + self.alpha * (backup_var - var)
                var.assign(new_backup_var)
                backup_var.assign(var)
