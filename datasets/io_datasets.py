import idx2numpy
from numpy import ndarray
from typing import Tuple


def get_mnist() -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    train_imgs = idx2numpy.convert_from_file("datasets/mnist/train_imgs.idx3-ubyte")
    train_labels = idx2numpy.convert_from_file("datasets/mnist/train_labels.idx1-ubyte")
    test_imgs = idx2numpy.convert_from_file("datasets/mnist/test_imgs.idx3-ubyte")
    test_labels = idx2numpy.convert_from_file("datasets/mnist/test_labels.idx1-ubyte")
    return train_imgs, train_labels, test_imgs, test_labels
