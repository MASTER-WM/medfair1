import numpy as np
from sklearn.metrics import confusion_matrix

def demographic_parity(labels, preds, protected_attrs):
    unique_attrs = np.unique(protected_attrs)
    dp = {}
    for attr in unique_attrs:
        idx = np.where(protected_attrs == attr)[0]
        dp[attr] = np.mean(preds[idx] == labels[idx])
    return dp

def equalized_odds(labels, preds, protected_attrs):
    unique_attrs = np.unique(protected_attrs)
    eo = {'TPR': {}, 'FPR': {}}
    for attr in unique_attrs:
        idx = np.where(protected_attrs == attr)[0]
        tn, fp, fn, tp = confusion_matrix(labels[idx], preds[idx]).ravel()
        eo['TPR'][attr] = tp / (tp + fn)
        eo['FPR'][attr] = fp / (fp + tn)
    return eo

def compute_fairness_loss(outputs, protected_attrs, coeff):
    # Example fairness loss function
    unique_attrs = np.unique(protected_attrs)
    fairness_loss = 0.0
    for attr in unique_attrs:
        idx = np.where(protected_attrs == attr)[0]
        fairness_loss += coeff * (outputs[idx].mean() - outputs.mean()).abs().sum()
    return fairness_loss