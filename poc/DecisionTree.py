import typing as t
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from poc.optimizable import OptimizableEstimator
from poc.learning_utils import load_train_test_split, Hyperparam


class DecisionTree(OptimizableEstimator):
    def __init__(self) -> None:
        super().__init__(
            DecisionTreeClassifier,
            [
                Hyperparam("min_samples_split"),
                Hyperparam("min_samples_leaf"),
                Hyperparam("min_weight_fraction_leaf"),
                Hyperparam("min_impurity_decrease"),
                Hyperparam("ccp_alpha"),
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

        self.clf = DecisionTreeClassifier()

    def compute_objective(self, x: np.ndarray) -> float:
        pass

    def get_x0(self) -> np.ndarray:
        raise NotImplementedError

    def get_constraints(self) -> t.List[dict]:
        raise NotImplementedError
