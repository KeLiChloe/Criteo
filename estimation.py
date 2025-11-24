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
        n_seg = idx.sum()
        
        if n_seg < 2:
            # Not enough samples in segment, set tau to 0
            raise ValueError(f"Segment {m} has less than 2 samples")
        
        # Get data for this segment
        X_m = X[idx]
        y_m = y[idx]
        D_m = D[idx]
        
        # Check if we have both treatments in this segment
        n_treated = (D_m == 1).sum()
        n_control = (D_m == 0).sum()
        
        if n_treated == 0 or n_control == 0:
            raise ValueError(f"Segment {m} has no treated or control samples")
        
        # Prepare design matrix: [1, X, D]
        # Shape: (n_seg, 1 + d + 1)
        X_design = np.column_stack([
            np.ones(n_seg),  # intercept
            X_m,             # features
            D_m              # treatment indicator
        ])
        
        # Fit linear regression: y = alpha + beta * X + tau * D
        reg = LinearRegression(fit_intercept=False)  # We already have intercept column
        reg.fit(X_design, y_m)
        
        # Extract tau (coefficient for D, which is the last column)
        tau_hat[m] = reg.coef_[-1]
        action[m] = 1 if tau_hat[m] > 0 else 0
    
    return tau_hat, action
