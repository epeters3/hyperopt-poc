import math

from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.metrics import mean_squared_error
from sklearn.svm import LinearSVR
import numpy as np

from poc.optimizable import OptimizableEstimator, Hyperparam, OptimizerValue


def rmse(y_true: np.ndarray, y_pred: np.ndarray):
    return math.sqrt(mean_squared_error(y_true, y_pred))


elasticnet_reg = OptimizableEstimator(
    ElasticNet(random_state=0, max_iter=1e7),
    optimizable_hyperparams=[
        Hyperparam("alpha", lbound=(0.0 + 1e-10)),
        Hyperparam("l1_ratio", lbound=0.01, ubound=1.0),
    ],
    score_func=rmse,
    # TODO: Add 2 or more artificial constraints for
    # the sake of the homework.
    # constraints=[
    #     {"type": "ineq", "fun": lambda x: x},
    #     {"type": "ineq", "fun": lambda x: x},
    # ],
    score_behavior=OptimizerValue(scale=100),
)
