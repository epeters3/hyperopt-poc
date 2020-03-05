import typing as t

import matplotlib.pyplot as plt
import numpy as np

from poc.optimizable_estimators import elasticnet_reg
from poc.visualize import create_hyperparam_grid, visualize_hyperparam_grid


def plot_circle(circle_eq: t.Callable, lbound: float, ubound: float) -> None:
    x = np.linspace(lbound, ubound, 100)
    y = np.linspace(lbound, ubound, 100)
    X, Y = np.meshgrid(x, y)
    F = circle_eq(X, Y)
    plt.contour(X, Y, F, [0])


def make_circle_into_constraint(circle: t.Callable) -> t.Callable:
    return lambda x: circle(x[0], x[1])


if __name__ == "__main__":
    DO_HOMEMADE = True
    USE_CONSTRAINTS = True

    if DO_HOMEMADE:
        method_to_use = "interior-penalty"
    else:
        method_to_use = "SLSQP"

    circle_a = lambda x, y: (x - 0.5) ** 2 + (y / 2) ** 2 - 0.1
    circle_b = lambda x, y: (x - 0.5) ** 2 + (y - 1.1) ** 2 - 0.1

    if USE_CONSTRAINTS:
        elasticnet_reg.constraints = [
            {"type": "ineq", "fun": make_circle_into_constraint(circle_a)},
            {"type": "ineq", "fun": make_circle_into_constraint(circle_b)},
        ]

    elasticnet_reg.set_dataset("boston")

    x_opts = []
    f_opts = []
    f_start = elasticnet_reg.score_behavior.from_optim(
        elasticnet_reg.compute_objective(elasticnet_reg.get_x0())
    )
    x_opts.append(elasticnet_reg.get_x0())
    f_opts.append(f_start)
    print("starting objective:", f_start)

    def cb(xk):
        f_opt = elasticnet_reg.score_behavior.from_optim(
            elasticnet_reg.compute_objective(xk)
        )
        x_opts.append(xk)
        f_opts.append(f_opt)
        print("objective:", f_opt)

    elasticnet_reg.verbose = True
    result = elasticnet_reg.optimize_hyperparams(method=method_to_use, callback=cb)
    grid = create_hyperparam_grid(
        elasticnet_reg, [((0.0 + 1e-10), 1.0), (0.01, 1.0)], 20
    )
    elasticnet_reg.verbose = False

    # visualize a grid of solutions
    sc = visualize_hyperparam_grid(elasticnet_reg, grid)

    if USE_CONSTRAINTS:
        # visualize the constrainsts as well
        plot_circle(circle_a, 0, 1)
        plot_circle(circle_b, 0, 1)

    # visualize the path the optimizer took
    x_1, x_2 = zip(*x_opts)
    plt.plot(x_1, x_2, c="r")
    plt.scatter(x_1[:-1], x_2[:-1], c="r", s=50)

    # mark the final optimum in the visual
    plt.scatter(x_1[-1], x_2[-1], marker="*", c="r", s=200)
    plt.savefig("optimization-run.png")
