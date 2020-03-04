import typing as t

import numpy as np
from scipy.optimize import OptimizeResult
import scipy as sp


def minimize(
    fun: t.Callable, x0: np.ndarray, method: str = None, **scipyargs
) -> OptimizeResult:
    """
    Wraps `scipy.optimize.minimize` to include custom
    interior penalty method.
    """
    if method == "interior-penalty":
        return interior_penalty(fun, x0, **scipyargs)
    else:
        return sp.optimize.minimize(fun, x0, method=method, **scipyargs)


def interior_penalty(
    fun: t.Callable,
    x0: np.ndarray,
    constraints: t.Sequence[dict] = (),
    min_mu: float = 1e-6,
    **scipyargs
) -> OptimizeResult:
    """
    A basic constrained optimizer that uses an interior penalty method to
    enforce the constraints. `fun`, `x0`, and `constraints` should all have the same
    types as `scipy.optimize.minimize(method="SLSQP")`. `min_mu` is the smallest value
    of the penalty parameter to go down to. Smaller increases the accuracy of the
    solution but takes longer. One optimization will be performed for each fractional
    magnitude in `min_mu`. It is assumed that all constraints in `constraints` are
    inequality constraints, of the form `h(x) >= 0`.
    """
    mu = 1.0

    def objective(x):
        return fun(x) - mu * np.sum(
            np.array([np.log(c["fun"](x)) for c in constraints])
        )

    fcalls = 0
    niters = 0
    while mu > min_mu:
        # Optimize the "unconstrained" penalized form of the objective.
        result = minimize(objective, x0, **scipyargs)
        x0 = result.x
        fcalls += result.nfev
        niters += result.nit
        mu /= 10

    result.nfev = fcalls
    result.nit = niters
    return result
