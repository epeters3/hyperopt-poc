from poc.optimizable_estimators import RidgeRegressor, DecisionTreeClassifier

if __name__ == "__main__":
    # regressor = RidgeRegressor()
    # cb = lambda xk: print(regressor.compute_objective(xk))
    # result = regressor.optimize_hyperparams(tol=1e-8, callback=cb)

    classifier = DecisionTreeClassifier()
    cb = lambda xk: print(classifier.compute_objective(xk))
    result = classifier.optimize_hyperparams(tol=1e-8, callback=cb)
