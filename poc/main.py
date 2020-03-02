from poc.optimizable_estimators import (
    RidgeRegressor,
    DecisionTreeClassifier,
    ElasticNetRegressor,
    RandomForestRegressor,
)

if __name__ == "__main__":
    regressor = ElasticNetRegressor()
    cb = lambda xk: print(regressor.compute_objective(xk))
    result = regressor.optimize_hyperparams(tol=1e-8, callback=cb)

    # classifier = DecisionTreeClassifier()
    # cb = lambda xk: print(classifier.compute_objective(xk))
    # result = classifier.optimize_hyperparams(tol=1e-8, callback=cb)
