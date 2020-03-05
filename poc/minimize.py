import typing as t

import numpy as np
from scipy.optimize import OptimizeResult
import scipy as sp


def minimize(
    fun: t.Callable, x0: np.ndarray, method: str = None, **scipyargs
) -> OptimizeResult:
    """
    Wraps `scipy.optimize.minimize` to include custom
    exterior penalty method.
    """
    if method == "exterior-penalty":
        return exterior_penalty(fun, x0, **scipyargs)
    else:
        return sp.optimize.minimize(fun, x0, method=method, **scipyargs)


def exterior_penalty(
    fun: t.Callable,
    x0: np.ndarray,
    constraints: t.Sequence[dict] = (),
    start_mu: float = 1,
    max_mu: float = 1e10,
    **scipyargs
) -> OptimizeResult:
    """
    A basic constrained optimizer that uses an exterior penalty method to
    enforce the constraints. `fun`, `x0`, and `constraints` should all have the same
    types as `scipy.optimize.minimize(method="SLSQP")`. `max_mu` is the largest value
    of the penalty parameter to go up to. Larger values increase the accuracy of the
    solution but takes longer. One optimization will be performed for each fractional
    magnitude in `max_mu`. It is assumed that all constraints in `constraints` are
    inequality constraints, of the form `h(x) >= 0`.
    """
    mu = start_mu

    def objective(x):
        return fun(x) + (mu / 2) * np.sum(
            np.array([min(0, c["fun"](x)) ** 2 for c in constraints])
        )

    fcalls = 0
    niters = 0
    while mu < max_mu:
        # Optimize the "unconstrained" penalized form of the objective.
        result = minimize(objective, x0, method="SLSQP", **scipyargs)
        x0 = result.x
        fcalls += result.nfev
        niters += result.nit
        mu *= 10

    result.nfev = fcalls
    result.nit = niters
    return result
