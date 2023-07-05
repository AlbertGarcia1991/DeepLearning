import os
import pickle
from typing import List, Tuple

import idx2numpy
import numpy as np
import pandas as pd
from numpy import ndarray


def get_mnist() -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    train_imgs = idx2numpy.convert_from_file("datasets/mnist/train_imgs.idx3-ubyte")
    train_labels = idx2numpy.convert_from_file("datasets/mnist/train_labels.idx1-ubyte")
    test_imgs = idx2numpy.convert_from_file("datasets/mnist/test_imgs.idx3-ubyte")
    test_labels = idx2numpy.convert_from_file("datasets/mnist/test_labels.idx1-ubyte")
    return train_imgs, train_labels, test_imgs, test_labels


def get_dataframe(data_iw: np.ndarray, gesture: str, idx_col: int) -> pd.DataFrame:
    df = pd.DataFrame(data_iw)
    df.columns = [
        "aX_L", "aY_L", "aZ_L", "gX_L", "gY_L", "gZ_L", "mX_L", "mY_L", "mZ_L",
        "aX_C", "aY_C", "aZ_C", "gX_C", "gY_C", "gZ_C", "mX_C", "mY_C", "mZ_C", "pressure",
        "aX_R", "aY_R", "aZ_R", "gX_R", "gY_R", "gZ_R", "mX_R", "mY_R", "mZ_R",
    ]
    df["gesture"] = gesture
    df["id"] = idx_col
    df = df.drop(
        columns=[
            "mX_L", "mY_L", "mZ_L",
            "aX_C", "aY_C", "aZ_C", "gX_C", "gY_C", "gZ_C", "mX_C", "mY_C", "mZ_C", "pressure",
            "mX_R", "mY_R", "mZ_R"
        ]
    )
    return df.reset_index(drop=True)


def get_gestures(gest_list: List[str], source: str = "all") -> Tuple[np.ndarray, np.ndarray]:
    data_path = "../datasets/quell_gestures/"
    data_files = os.listdir(data_path)

    data_files_source_observer = []
    data_files_source_old = []
    if source == "all" or source == "observer":
        data_files_source_observer = [
            file for file in data_files if "OBSERVER" in file and file.split("_")[2] in gest_list
        ]
    if source == "all" or source == "old":
        data_files_source_old = [
            file for file in data_files if "CROP" in file and file.split("_")[2] in gest_list
        ]
    data_files = data_files_source_observer + data_files_source_old
    data_raw = {}
    for file in data_files:
        with open(os.path.join(data_path, file), "rb") as f:
            data = pickle.load(f)[1]
        if file.split("_")[2] not in data_raw.keys():
            data_raw[file.split("_")[2]] = [data]
        else:
            data_raw[file.split("_")[2]].append(data)

    idx_col = 0
    for gesture in gest_list:
        for data in data_raw[gesture]:
            for data_iw in data:
                if idx_col == 0:
                    df = get_dataframe(data_iw=data_iw, gesture=gesture, idx_col=idx_col)
                else:
                    df = pd.concat((df, get_dataframe(data_iw=data_iw, gesture=gesture, idx_col=idx_col)))
                idx_col += 1

    df = df.loc[
         :,
         [
             "gesture", "id",
             "aX_L", "aY_L", "aZ_L", "gX_L", "gY_L", "gZ_L",
             "aX_R", "aY_R", "aZ_R", "gX_R", "gY_R", "gZ_R"
         ]
    ]

    X, y = [], []
    gesture_int_map = {}
    for id_ in df.id.unique():
        X.append(df[df.id == id_].iloc[:, 2:])
        gesture = df[df.id == id_].iloc[0, 0]
        if gesture not in gesture_int_map.keys():
            gesture_int_map[gesture] = len(gesture_int_map)
        y.append(gesture_int_map[gesture])
    X, y = np.array(X), np.array(y)

    return X, y
