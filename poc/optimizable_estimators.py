import math

from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score, mean_squared_error

from poc.optimizable import OptimizableEstimator, Hyperparam, OptimizerValue


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
            # RMSE
            score_func=lambda y_true, y_pred: math.sqrt(
                mean_squared_error(y_true, y_pred)
            ),
            score_behavior=OptimizerValue(scale=100),
        )
