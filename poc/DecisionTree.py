import typing as t
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from poc.optimizable import OptimizableEstimator
from poc.learning_utils import load_train_test_split, Hyperparam


class DecisionTree(OptimizableEstimator):
    def __init__(self) -> None:
        super().__init__(
            DecisionTreeClassifier(random_state=0),
            [
                Hyperparam("min_samples_split", is_int=True, scale=100, lbound=1),
                Hyperparam("min_samples_leaf", is_int=True, scale=100, lbound=1),
                Hyperparam("min_weight_fraction_leaf", lbound=0.0, ubound=1.0),
                Hyperparam("min_impurity_decrease", lbound=0.0, ubound=1.0),
                Hyperparam("ccp_alpha", lbound=0.0),
            ],
        )
        train, test = load_train_test_split("iris")
        self.train = train
        self.test = test

        self.hyperparam2index = {
            name: i for i, name in enumerate(self.optimizable_params)
        }
        self.index2hyperparam = {
            i: name for i, name in enumerate(self.optimizable_params)
        }

    def compute_objective(self, x: np.ndarray) -> float:
        pass

    def get_x0(self) -> np.ndarray:
        raise NotImplementedError

    def get_constraints(self) -> t.List[dict]:
        raise NotImplementedError
