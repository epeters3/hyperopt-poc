import typing as t
import itertools

import numpy as np
from matplotlib import pyplot as plt

from poc.optimizable import OptimizableEstimator
from poc.optimizable_estimators import elasticnet_reg


def create_hyperparam_grid(
    est: OptimizableEstimator, bounds: t.Sequence[t.Tuple[float, float]], n: int
) -> np.ndarray:
    grids_by_dim = [np.linspace(lbound, ubound, n) for lbound, ubound in bounds]
    full_grid_points = itertools.product(*grids_by_dim)
    return list(full_grid_points)


def visualize_hyperparam_grid(
    est: OptimizableEstimator, grid: t.Iterable[t.Tuple[float, ...]]
) -> None:
    if len(est.optimizable_hyperparams) != 2:
        raise ValueError("can only visualize models with two optimizable hyperparams.")

    objectives = []
    for x in grid:
        scaled_x = [
            h_i.to_optim(x_i) for x_i, h_i in zip(x, est.optimizable_hyperparams)
        ]
        objectives.append(
            est.score_behavior.from_optim(est.compute_objective(scaled_x))
        )

    x, y = zip(*grid)
    sc = plt.scatter(x, y, c=objectives)
    plt.xlabel(est.optimizable_hyperparams[0].name)
    plt.ylabel(est.optimizable_hyperparams[1].name)
    plt.title(f"{est.est.__class__.__name__} Grid Search")
    plt.colorbar(sc)
    return sc


if __name__ == "__main__":
    elasticnet_reg.set_dataset("boston")
    grid = create_hyperparam_grid(
        elasticnet_reg, [((0.0 + 1e-10), 5.0), (0.01, 1.0)], 20
    )
    visualize_hyperparam_grid(elasticnet_reg, grid)
