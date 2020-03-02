from poc.optimizable_estimators import (
    ridge_reg,
    decision_tree_clf,
    elasticnet_reg,
    rf_reg,
    huber_reg,
    sv_reg,
)

if __name__ == "__main__":
    result = elasticnet_reg.optimize_hyperparams(
        "boston",
        tol=1e-8,
        callback=lambda xk: print(
            "raw objective:", elasticnet_reg.compute_objective(xk)
        ),
    )
