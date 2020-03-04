import math

from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.svm import LinearSVR
import numpy as np

from poc.optimizable import OptimizableEstimator, Hyperparam, OptimizerValue


def rmse(y_true: np.ndarray, y_pred: np.ndarray):
    return math.sqrt(mean_squared_error(y_true, y_pred))


def f1_macro(y_true: np.ndarray, y_pred: np.ndarray):
    return f1_score(y_true, y_pred, average="macro")


decision_tree_clf = OptimizableEstimator(
    DT(random_state=0),
    optimizable_hyperparams=[
        Hyperparam("min_samples_split", is_int=True, scale=100, lbound=1),
        Hyperparam("min_samples_leaf", is_int=True, scale=100, lbound=1),
        Hyperparam("min_weight_fraction_leaf", lbound=0.0, ubound=1.0),
        Hyperparam("min_impurity_decrease", lbound=0.0, ubound=1.0),
        Hyperparam("ccp_alpha", lbound=0.0),
    ],
    score_func=f1_macro,
    # Scale is negative because we want to maximize the score.
    score_behavior=OptimizerValue(scale=-1),
)


ridge_reg = OptimizableEstimator(
    Ridge(random_state=0),
    optimizable_hyperparams=[Hyperparam("alpha", lbound=0.0)],
    score_func=rmse,
    score_behavior=OptimizerValue(scale=100),
)


elasticnet_reg = OptimizableEstimator(
    ElasticNet(random_state=0, max_iter=1e7),
    optimizable_hyperparams=[
        Hyperparam("alpha", lbound=(0.0 + 1e-10)),
        Hyperparam("l1_ratio", lbound=0.01, ubound=1.0),
    ],
    score_func=rmse,
    # TODO: Add 2 or more artificial constraints for
    # the sake of the homework.
    score_behavior=OptimizerValue(scale=100),
)


rf_reg = OptimizableEstimator(
    RFR(random_state=0),
    optimizable_hyperparams=[
        Hyperparam("min_weight_fraction_leaf", lbound=0.0, ubound=1.0),
        Hyperparam("min_impurity_decrease", lbound=0.0, ubound=1.0),
        Hyperparam("ccp_alpha", lbound=0.0, ubound=1.0),
    ],
    score_func=rmse,
    score_behavior=OptimizerValue(scale=100),
)

huber_reg = OptimizableEstimator(
    HuberRegressor(max_iter=int(1e6)),
    optimizable_hyperparams=[
        Hyperparam("epsilon", lbound=(1 + 1e-10)),  # > 1
        Hyperparam("alpha", lbound=0.0, scale=1e-4),
    ],
    score_func=rmse,
    score_behavior=OptimizerValue(scale=100),
)

sv_reg = OptimizableEstimator(
    LinearSVR(max_iter=int(1e6)),
    optimizable_hyperparams=[
        Hyperparam("epsilon", lbound=0.0),  # > 1
        Hyperparam("C", lbound=(0.0 + 1e-10)),  # > 0
    ],
    score_func=rmse,
    score_behavior=OptimizerValue(scale=100),
)
