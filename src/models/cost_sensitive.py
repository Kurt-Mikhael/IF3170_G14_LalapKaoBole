from numbers import Real

import numpy as np

# returns the bayes-optimal positive-class threshold for a cost matrix
# a positive prediction is selected when P(y=1|x) >= C_FP / (C_FP + C_FN)
def cost_sensitive_threshold(false_positive_cost=2.0, false_negative_cost=1.0):
    for name, value in (("false_positive_cost", false_positive_cost), ("false_negative_cost", false_negative_cost)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError(f"{name} must be a finite positive number")
        if value <= 0:
            raise ValueError(f"{name} must be a positive number")
    
    return float(false_positive_cost / (false_positive_cost + false_negative_cost))

# classify positive probabilities under the supplied error costs
def predict_cost_sensitive(probability_positive, false_positive_cost=2.0, false_negative_cost=1.0):
    probabilities = np.asarray(probability_positive, dtype=float)
    
    if probabilities.ndim != 1 or not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probability_positive must be a finite one-dimensional array in [0, 1]")
    threshold = cost_sensitive_threshold(false_positive_cost, false_negative_cost)
    
    return (probabilities >= threshold).astype(int)

# return false-positive count, false-negative count, and total error cost
def classification_cost(y_true, y_pred, false_positive_cost=2.0, false_negative_cost=1.0):
    threshold = cost_sensitive_threshold(false_positive_cost, false_negative_cost)
    del threshold  # validates both costs in one place
    actual = np.asarray(y_true)
    predicted = np.asarray(y_pred)

    if actual.ndim != 1 or predicted.ndim != 1 or len(actual) != len(predicted):
        raise ValueError("y_true and y_pred must be one-dimensional arrays of equal length")
    
    if not np.isin(actual, [0, 1]).all() or not np.isin(predicted, [0, 1]).all():
        raise ValueError("y_true and y_pred must contain only binary labels 0 and 1")
    false_positive = int(np.sum((actual == 0) & (predicted == 1)))
    false_negative = int(np.sum((actual == 1) & (predicted == 0)))
    total = false_positive_cost * false_positive + false_negative_cost * false_negative
    
    return {"false_positive": false_positive, "false_negative": false_negative, "total_cost": float(total)}