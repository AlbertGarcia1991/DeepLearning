import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

from datasets.io_datasets import get_gestures
from utils.layer import *
from utils.model import *
from utils.op_tensor import *
from utils.optimizer import *
from utils.metric import *
from utils.loss import *
from utils.train import train_model


def preprocess(
        X_train: ndarray,
        y_train: ndarray,
        X_test: ndarray,
        y_test: ndarray,
        n_classes: int
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    X_train = op_flatten_input(X=X_train)
    X_test = op_flatten_input(X=X_test)
    y_train = op_one_hot(y=y_train, n_classes=n_classes)
    y_test = op_one_hot(y=y_test, n_classes=n_classes)

    return X_train, y_train, X_test, y_test


# 1) Load quell_gestures
X, y = get_gestures(gest_list=["BHO", "BHC"], source="all")

# 2) Window selection and processing
w_start = -20
w_size = 40
X = X[:, 52 + w_start:52 + w_start + w_size, :]
N_classes = len(np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42
)
X_train, y_train, X_test, y_test = preprocess(
    X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, n_classes=N_classes
)
train_Xy = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_Xy = tf.data.Dataset.from_tensor_slices((X_test, y_test))

N_examples, N_features = X_train.shape


# 3) Create dummy model
dummy_model = ModelMLP(
    layers=[
        DenseLayer(out_dims=int(2 / 3 * N_features + N_classes)),
        DenseLayer(out_dims=N_classes, soft_max_flag=True),
    ]
)

train_losses, train_accs, val_losses, val_accs = train_model(
    model=dummy_model,
    train_Xy=train_Xy,
    val_Xy=test_Xy,
    loss=loss_binary_cross_entropy,
    acc=metric_accuracy,
    optimizer=OptimizerAdam(),
    batch_size=16,
    epochs=100,
    early_stop=["acc", 1520, 1]
)

# 4) Find better preprocessing technique

# 5) Tune topology of a new model applying the best preprocessing technique from 4)

"""
1) Set dummy network
	- ADAM optimizer
	- ReLU activation
	- He initialization
	- 1 hidden layer: N_h = N_i * (2 / 3) + N_o
------------------------ START SEARCHING ALGORITHM (MLP)
2) Benchmark preprocessing techniques
3) Benchmark Topology -> CONSTRAINTS: Size and Latency
4) Benchmark Weight Initialization
5) Benchmark Batch Normalization
6) Benchmark Dropout
7) Benchmark LR
8) Benchmark Epochs & Batch Size
9) Benchmark L1/L2 Regularization
10) Benchmark Activation Function
----------------------- END SEARCHING ALGORITHM
11) Final training using all data
12) Serialize
13) Deploy
14) Log (MLflow?)

------------------------ START SEARCHING ALGORITHM (ResNET)
------------------------ START SEARCHING ALGORITHM (MLSTM_FCN)
"""