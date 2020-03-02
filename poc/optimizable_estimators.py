import math

from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.metrics import f1_score, mean_squared_error
import numpy as np

from poc.optimizable import OptimizableEstimator, Hyperparam, OptimizerValue


def rmse(y_true: np.ndarray, y_pred: np.ndarray):
    return math.sqrt(mean_squared_error(y_true, y_pred))


class DecisionTreeClassifier(OptimizableEstimator):
    def __init__(self) -> None:
        super().__init__(
            DT(random_state=0),
            optimizable_hyperparams=[
                Hyperparam("min_samples_split", is_int=True, scale=100, lbound=1),
                Hyperparam("min_samples_leaf", is_int=True, scale=100, lbound=1),
                Hyperparam("min_weight_fraction_leaf", lbound=0.0, ubound=1.0),
                Hyperparam("min_impurity_decrease", lbound=0.0, ubound=1.0),
                Hyperparam("ccp_alpha", lbound=0.0),
            ],
            dataset_name="iris",
            score_func=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"),
            # Scale is negative because we want to maximize the score.
            score_behavior=OptimizerValue(scale=-1),
        )


class RidgeRegressor(OptimizableEstimator):
    def __init__(self) -> None:
        super().__init__(
            Ridge(random_state=0),
            optimizable_hyperparams=[Hyperparam("alpha", lbound=0.0)],
            dataset_name="boston",
            score_func=rmse,
            score_behavior=OptimizerValue(scale=100),
        )


class ElasticNetRegressor(OptimizableEstimator):
    def __init__(self) -> None:
        super().__init__(
            ElasticNet(random_state=0, fit_intercept=True, max_iter=1e6),
            optimizable_hyperparams=[
                Hyperparam("alpha", lbound=0.0),
                Hyperparam("l1_ratio", lbound=0.0, ubound=1.0),
            ],
            dataset_name="boston",
            score_func=rmse,
            score_behavior=OptimizerValue(scale=100),
        )


class RandomForestRegressor(OptimizableEstimator):
    def __init__(self) -> None:
        super().__init__(
            RFR(random_state=0),
            optimizable_hyperparams=[
                Hyperparam("min_weight_fraction_leaf", lbound=0.0, ubound=1.0),
                Hyperparam("min_impurity_decrease", lbound=0.0, ubound=1.0),
                Hyperparam("ccp_alpha", lbound=0.0, ubound=1.0),
            ],
            dataset_name="boston",
            score_func=rmse,
            score_behavior=OptimizerValue(scale=100),
        )
