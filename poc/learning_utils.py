import typing as t

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_boston


class Dataset(t.NamedTuple):
    X: np.ndarray
    y: np.ndarray


class Hyperparam(t.NamedTuple):
    name: str
    lower_bound: float
    upper_bound: float
    is_integer: bool
    scaling_factor: float


def sk_data_to_pd(dataset: dict) -> t.Tuple[np.ndarray, np.ndarray]:
    X = dataset["data"]
    y = dataset["target"]
    return X, y


def load_dataset(name: str) -> t.Tuple[np.ndarray, np.ndarray]:
    loaders: t.Dict[str, t.Callable] = {
        "iris": lambda: sk_data_to_pd(load_iris()),  # classification
        "boston": lambda: sk_data_to_pd(load_boston()),  # regression
    }
    assert name in loaders
    return loaders[name]()


def load_train_test_split(
    name: str, test_size: float = 0.33
) -> t.Tuple[Dataset, Dataset]:
    """
    Returns train/test split of dataset identified
    by `name`.
    """
    X, y = load_dataset(name)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size, random_state=0)
    return Dataset(X_train, y_train), Dataset(X_test, y_test)
