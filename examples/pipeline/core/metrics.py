"""Default metrics by task type. GENERIC — a track only overrides ``Track.metric``
if it needs something custom (VRMSE multi-step rollout, per-frame macro-AUROC, ...)."""
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             mean_absolute_error, mean_squared_error, roc_auc_score)


def classification_metrics(y, prob, n_classes):
    """y: [N] int. prob: [N] (binary, P(class=1)) or [N, K] (multiclass)."""
    if n_classes == 2:
        pred = (prob >= 0.5).astype(int)
        return {"acc": accuracy_score(y, pred), "bal_acc": balanced_accuracy_score(y, pred),
                "f1": f1_score(y, pred), "auroc": roc_auc_score(y, prob)}
    pred = prob.argmax(1) if prob.ndim == 2 else prob
    return {"acc": accuracy_score(y, pred), "bal_acc": balanced_accuracy_score(y, pred),
            "f1_macro": f1_score(y, pred, average="macro")}


def regression_metrics(y, pred):
    """y, pred: [N] floats (e.g. next-step return)."""
    corr = float(np.corrcoef(y, pred)[0, 1]) if np.std(pred) > 1e-9 else 0.0
    return {"mse": mean_squared_error(y, pred), "mae": mean_absolute_error(y, pred),
            "dir_acc": float((np.sign(y) == np.sign(pred)).mean()), "corr": corr}


def forecasting_metrics(y, pred):
    """y, pred: [N, C, H] multivariate forecast (already normalized)."""
    return {"mse": float(np.mean((y - pred) ** 2)), "mae": float(np.mean(np.abs(y - pred)))}


def default_metric(task_type, n_classes=2):
    return {"classification": lambda y, p: classification_metrics(y, p, n_classes),
            "regression": regression_metrics,
            "forecasting": forecasting_metrics}[task_type]
