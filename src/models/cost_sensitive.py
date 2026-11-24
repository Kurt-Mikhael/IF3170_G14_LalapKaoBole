from numbers import Real

import numpy as np

# ambang kelas positif yang optimal untuk biaya kesalahan tertentu
# prediksi positif dipilih jika P(y=1|x) >= C_FP / (C_FP + C_FN)
def cost_sensitive_threshold(false_positive_cost=2.0, false_negative_cost=1.0):
    for name, value in (("false_positive_cost", false_positive_cost), ("false_negative_cost", false_negative_cost)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError(f"{name} must be a finite positive number")
        if value <= 0:
            raise ValueError(f"{name} must be a positive number")
    
    return float(false_positive_cost / (false_positive_cost + false_negative_cost))

# memprediksi kelas dari peluang positif dengan ambang berdasarkan biaya
def predict_cost_sensitive(probability_positive, false_positive_cost=2.0, false_negative_cost=1.0):
    probabilities = np.asarray(probability_positive, dtype=float)
    
    if probabilities.ndim != 1 or not np.isfinite(probabilities).all():
        raise ValueError("probability_positive must be a finite one-dimensional array in [0, 1]")
    # selisih pembulatan kecil (misalnya 1+1e-15 dari proporsi daun pohon) dipotong ke [0, 1]
    if np.any((probabilities < -1e-9) | (probabilities > 1 + 1e-9)):
        raise ValueError("probability_positive must be a finite one-dimensional array in [0, 1]")
    probabilities = np.clip(probabilities, 0.0, 1.0)
    threshold = cost_sensitive_threshold(false_positive_cost, false_negative_cost)
    
    return (probabilities >= threshold).astype(int)

# mengembalikan jumlah false positive, false negative, dan total biaya
def classification_cost(y_true, y_pred, false_positive_cost=2.0, false_negative_cost=1.0):
    threshold = cost_sensitive_threshold(false_positive_cost, false_negative_cost)
    del threshold  # hanya untuk memvalidasi kedua biaya
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