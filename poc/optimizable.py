from abc import ABC, abstractmethod
import typing as t

import numpy as np
from sklearn.base import BaseEstimator
from scipy.optimize import minimize, OptimizeResult

from poc.learning_utils import load_train_test_split


class OptimizerValue:
    """
    Provides methods and and holds metadata related
    to interacting with values that will
    be used by a gradient-based optimizer. Handles
    scaling and smoothing so the optimizer can have
    more continuous values that are of O(1). The `scale`
    attribute should be set appropriately so the value
    is of O(1) (i.e. in general ranges from [0,1]).
    """

    def __init__(self, *, is_int: bool = False, scale: float = 1.0,) -> None:
        self.is_int = is_int
        self.scale = scale

    def to_optim(self, value: float) -> float:
        """
        Converts `value` from its representation used in the wild
        outside an optimizer to its representation used by
        the optimizer (i.e. its scaled version).
        """
        return value / self.scale

    def from_optim(self, value: float) -> float:
        """
        Converts `value` from its representation used by an optimizer
        to its representation used in in the wild (i.e. its raw, unscaled,
        unsmoothed version).
        """
        de_scaled = value * self.scale
        if self.is_int:
            de_scaled = int(round(de_scaled))
        return de_scaled


class Hyperparam(OptimizerValue):
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
        An optional lower bound for this hyperparameter. The value
        should be expressed in the parameter's original scale.
    ubound:
        An optional upper bound for this hyperparameter. The value
        should be expressed in the parameter's original scale.
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
        super().__init__(is_int=is_int, scale=scale)
        self.name = name
        self.lbound = lbound
        self.ubound = ubound

    # @abstractmethodz
    # def get_constraints(self) -> t.List[dict]:
    #     """
    #     Returns the constraints this hyperparameter
    #     requires when being optimized by `scipy.optimize.minimize`.
    #     """
    #     # TODO
    #     raise NotImplementedError


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
    def get_bounds(self) -> t.Sequence[t.Tuple[float, float]]:
        """
        Should return the values to pass to the `bounds`
        argument of `scipy.optimize.minimize`
        (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
        """
        raise NotImplementedError

    # @abstractmethod
    # def get_constraints(self) -> t.Sequence[dict]:
    #     """
    #     Should return the values to pass to the `constraints`
    #     argument of `scipy.optimize.minimize`
    #     (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
    #     """
    #     # TODO: Make Sub-classes implement this.
    #     raise NotImplementedError


class OptimizableEstimator(Optimizable):
    def __init__(
        self,
        est: BaseEstimator,
        *,
        optimizable_hyperparams: t.Sequence[Hyperparam],
        dataset_name: str,
        score_func: t.Callable[[np.ndarray, np.ndarray], float],
        score_behavior: OptimizerValue,
    ) -> None:
        """
        Parameters
        ----------
        est:
            The instantiated sklearn estimator
        optimizable_hyperparams:
            The list of hyper parameter objects representing the
            hyperparameters this estimator can be optimized with.
        dataset_name:
            The name of the dataset to optimize this estimator on.
        score_func:
            Should be of the form `score = score_func(y_true, y_pred)`.
        score_behavior:
            Information about how to scale the score before passing it
            to the optimizer so that values inside the optimzer can all
            be well-behaved and of a similar magnitude. If the score of
            this estimator needs to be minimized, include a negative in
            the behavior.
        """
        self.est = est
        self.optimizable_hyperparams = optimizable_hyperparams
        self.score_func = score_func
        self.score_behavior = score_behavior

        # be able to get hyperparam by name
        self.hyperparamindex = {
            hp.name: i for i, hp in enumerate(self.optimizable_hyperparams)
        }

        self.dataset_name = dataset_name
        train_data, val_data = load_train_test_split(self.dataset_name)
        self.train_data = train_data
        self.val_data = val_data

    def get_x0(self) -> np.ndarray:
        """
        Returns the params as set in the sklearn estimator, converted
        to the scaled version the optimizer will use.
        """
        sk_params = self.est.get_params()
        x0 = [hp.to_optim(sk_params[hp.name]) for hp in self.optimizable_hyperparams]
        return np.array(x0)

    def get_bounds(self) -> t.Sequence[t.Tuple[float, float]]:
        return [
            (
                None if hp.lbound is None else hp.to_optim(hp.lbound),
                None if hp.ubound is None else hp.to_optim(hp.ubound),
            )
            for hp in self.optimizable_hyperparams
        ]

    def from_optim(self, x: np.ndarray) -> dict:
        """
        Takes the hyperparameter values as the optimizer uses
        them and reforms them into the form sklearn uses.
        """
        # descale the `x` vector incoming from the optimizer.
        x_for_model = [
            hp_i.from_optim(x_i) for x_i, hp_i in zip(x, self.optimizable_hyperparams)
        ]
        # Map each x to the name of the hyperparameter it goes to.
        x_by_name = {
            hp_i.name: x_i
            for x_i, hp_i in zip(x_for_model, self.optimizable_hyperparams)
        }
        print(x_by_name)
        return x_by_name

    def compute_objective(self, x: np.ndarray) -> float:
        reformed_x = self.from_optim(x)
        self.est.set_params(**reformed_x)
        self.est.fit(self.train_data.X, self.train_data.y)
        y_pred = self.est.predict(self.val_data.X)
        score = self.score_func(self.val_data.y, y_pred)
        # This is a model score. Scale it so the optimizer
        # can work with a problem that's not ill-conditioned.
        return self.score_behavior.to_optim(score)

    def optimize_hyperparams(self, **optimizerargs) -> OptimizeResult:
        """
        Performs hyperparameter optimization on `problem`. Passes
        `optimizerargs` on to the optimizer.
        """
        result = minimize(
            fun=self.compute_objective,
            x0=self.get_x0(),
            bounds=self.get_bounds(),
            **optimizerargs,
        )
        # We need to de-scale the resulting x vector and
        # score (objective).
        result.x = self.from_optim(result.x)
        result.fun = self.score_behavior.from_optim(result.fun)
        self.print_hyperopt_result(result)
        return result

    def print_hyperopt_result(self, result: OptimizeResult) -> None:
        print("\nOptimization Results")
        print("--------------------")
        print(f"x_opt\t:\t{result.x}")
        print(f"success\t:\t{result.success}")
        print(f"status\t:\t{result.status}")
        print(f"message\t:\t{result.message}")
        print(f"f_opt\t:\t{result.fun}")
        print(f"gradient\t:\t{result.jac}")
        print(f"fcalls\t:\t{result.nfev}")
        print(f"niters\t:\t{result.nit}")
