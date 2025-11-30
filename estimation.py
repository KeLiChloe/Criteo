# estimation.py

import numpy as np
from sklearn.linear_model import LinearRegression


def estimate_segment_policy(X, y, D, seg_labels):
    """
    Estimate segment-level policy using regression: y = alpha + beta * X + tau * D
    
    For each segment m, fit: y_m = alpha_m + beta_m * X_m + tau_m * D_m
    where tau_m is the treatment effect for segment m.
    
    Parameters
    ----------
    X : np.ndarray, shape (N, d)
        Feature matrix
    y : np.ndarray, shape (N,)
        Outcome vector
    D : np.ndarray, shape (N,)
        Treatment assignment (0 or 1)
    seg_labels : np.ndarray, shape (N,)
        Segment assignments for each sample
    
    Returns
    -------
    tau_hat : np.ndarray
        Estimated treatment effect (tau) for each segment
    action : np.ndarray
        Recommended action (0 or 1) for each segment
    """
    M = int(seg_labels.max() + 1)
    tau_hat = np.zeros(M)
    action = np.zeros(M, dtype=int)
    
    for m in range(M):
        idx = (seg_labels == m)
        
        # use the diffs in means (y) to estimate tau_hat
        y_seg = y[idx]
        y_0 = y_seg[D[idx] == 0]
        y_1 = y_seg[D[idx] == 1]
        if len(y_0) == 0 or len(y_1) == 0:
            action[m] = np.random.choice([0, 1])
        else:
            tau_hat[m] = y_1.mean() - y_0.mean()
            action[m] = 1 if tau_hat[m] > 0 else 0
    
    return action
