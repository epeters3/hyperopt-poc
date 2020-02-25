from abc import ABC, abstractmethod
import typing as t

import numpy as np

from poc.learning_utils import Hyperparam


class Optimizable(ABC):
    """
    A class implementing this interface can be optimized by the
    `scipy.optimize.minimize` method.
    """

    @abstractmethod
    def compute_objective(self, x: np.ndarray) -> float:
        """
        Returns the objective of this optimizable thing. The
        optimizer will attempt to minimize this. `x` are the
        design variables
        """
        raise NotImplementedError

    @abstractmethod
    def get_x0(self) -> np.ndarray:
        """
        Get this optimizable's initial guess for the design
        variables. The optimizer will use it as it's starting
        point.
        """
        raise NotImplementedError

    @abstractmethod
    def get_constraints(self) -> t.List[dict]:
        """
        Should return the values to pass to the `constraints`
        argument of `scipy.optimize.minimize`
        (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
        """
        raise NotImplementedError


class OptimizableEstimator(Optimizable):
    def __init__(
        self, clf_cls: t.Type, optimizable_params: t.Sequence[Hyperparam]
    ) -> None:
        self.clf = clf_cls
        self.optimizable_params = optimizable_params

    def get_x0(self) -> np.ndarray:
        params = self.clf.get_params()
        return np.ndarray([params[name] for name in self.optimizable_params])
