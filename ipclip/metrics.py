import numpy as np
from sklearn.metrics import auc, roc_curve, f1_score, recall_score, precision_score, matthews_corrcoef, accuracy_score, classification_report

def retrieval_metrics(y_target, y_predictions):
    p_10 = sum(int(t in preds[:10]) for t, preds in zip(y_target, y_predictions))
    p_50 = sum(int(t in preds[:50]) for t, preds in zip(y_target, y_predictions))
    n = len(y_target)
    return {"p@10": p_10 / n, "p@50": p_50 / n}

def eval_metrics(y_true, y_pred, y_pred_proba=None, average_method='weighted'):
    assert len(y_true) == len(y_pred)
    if y_pred_proba is None or len(np.unique(y_true)) > 2:
        auroc = np.nan
    else:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auroc = auc(fpr, tpr)
    f1 = f1_score(y_true, y_pred, average=average_method)
    precision = precision_score(y_true, y_pred, average=average_method)
    recall = recall_score(y_true, y_pred, average=average_method)
    mcc = matthews_corrcoef(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    # confusion components (binary-friendly)
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yp == 1 and yt != yp)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yp == 0 and yt != yp)
    sensitivity = (tp / (tp + fn)) if (tp + fn) else np.nan
    specificity = (tn / (tn + fp)) if (tn + fp) else np.nan
    ppv = (tp / (tp + fp)) if (tp + fp) else np.nan
    npv = (tn / (tn + fn)) if (tn + fn) else np.nan
    hitrate = (tp + tn) / max(1, (tp + tn + fp + fn))
    return {
        'Accuracy': acc, 'AUC': auroc, 'WF1': f1,
        'precision': precision, 'recall': recall, 'mcc': mcc,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'sensitivity': sensitivity, 'specificity': specificity,
        'ppv': ppv, 'npv': npv, 'hitrate': hitrate, 'instances': len(y_true)
    }
