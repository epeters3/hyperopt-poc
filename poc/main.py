from poc.optimizable_estimators import (
    ridge_reg,
    decision_tree_clf,
    elasticnet_reg,
    rf_reg,
    huber_reg,
    sv_reg,
)

if __name__ == "__main__":
    do_homemade = True
    if do_homemade:
        method_to_use = "interior-penalty"
    else:
        method_to_use = "SLSQP"
    elasticnet_reg.set_dataset("boston")
    print(
        "starting objective:",
        elasticnet_reg.score_behavior.from_optim(
            elasticnet_reg.compute_objective(elasticnet_reg.get_x0())
        ),
    )
    result = elasticnet_reg.optimize_hyperparams(
        method=method_to_use,
        callback=lambda xk: print(
            "objective:",
            elasticnet_reg.score_behavior.from_optim(
                elasticnet_reg.compute_objective(xk)
            ),
        ),
    )
