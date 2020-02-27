import typing as t

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_boston


class Dataset(t.NamedTuple):
    X: np.ndarray
    y: np.ndarray


class Hyperparam:
    """
    Provides methods and and holds metadata related
    to interacting with hyperparameters that will
    be optimized by a gradient-based optimizer.

    Attributes
    ----------
    name:
        The `sklearn` name of the hyperparameter.
    is_int:
        Whether `sklearn` requires that this hyperparameter
        be an integer. Defaults to `False`.
    lbound:
        An optional lower bound for this hyperparameter.
    ubound:
        An optional upper bound for this hyperparameter.
    scale:
        The number to scale `sklearn`'s value of this
        hyperparameter by to ensure that the optimizer's
        representation of this hyperparameter is approximately
        in the range `[0,1]`.
    """

    def __init__(
        self,
        name: str,
        *,
        is_int: bool = False,
        lbound: float = None,
        ubound: float = None,
        scale: float = 1.0,
    ) -> None:
        self.name = name
        self.lbound = lbound
        self.ubound = ubound
        self.is_int = is_int
        self.scale = scale

    def to_diff(self, model_val: float) -> float:
        """
        Converts a value for this hyperparameter from its
        representation used in an sklearn model to its
        representation used by an optimizer (i.e. its
        scaled, smoothed version).
        """
        return model_val / self.scale

    def to_model(self, diff_val: float) -> float:
        """
        Converts a value for this hyperparameter from its
        representation used by an optimizer to its
        representation used in an sklearn modell.
        """
        de_scaled = diff_val * self.scale
        if self.is_int:
            de_scaled = round(de_scaled)
        return de_scaled

    def get_constraints(self) -> t.List[dict]:
        """
        Returns the constraints this hyperparameter
        requires when being optimized by `scipy.optimize.minimize`.
        """
        # TODO
        raise NotImplementedError


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
