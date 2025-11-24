# data_utils.py
from sklearn.model_selection import train_test_split

def split_pilot_impl(X, D, y, pilot_frac=0.7, random_state=0):
    """
    Full data → pilot + implementation
    """
    X_pilot, X_impl, D_pilot, D_impl, y_pilot, y_impl = train_test_split(
        X, D, y,
        train_size=pilot_frac,
        random_state=random_state,
        stratify=D
    )
    
    # Convert to numpy arrays if they are pandas objects
    # This ensures consistent integer-based indexing (not label-based)
    # Note: After train_test_split, indices may be non-contiguous, so we reset first
    if hasattr(X_pilot, 'reset_index'):
        X_pilot = X_pilot.reset_index(drop=True).values
        X_impl = X_impl.reset_index(drop=True).values
    elif hasattr(X_pilot, 'values'):
        X_pilot = X_pilot.values
        X_impl = X_impl.values
    
    if hasattr(D_pilot, 'reset_index'):
        D_pilot = D_pilot.reset_index(drop=True).values
        D_impl = D_impl.reset_index(drop=True).values
    elif hasattr(D_pilot, 'values'):
        D_pilot = D_pilot.values
        D_impl = D_impl.values
    
    if hasattr(y_pilot, 'reset_index'):
        y_pilot = y_pilot.reset_index(drop=True).values
        y_impl = y_impl.reset_index(drop=True).values
    elif hasattr(y_pilot, 'values'):
        y_pilot = y_pilot.values
        y_impl = y_impl.values
    
    return X_pilot, X_impl, D_pilot, D_impl, y_pilot, y_impl


def split_seg_train_test(X_pilot, D_pilot, y_pilot,
                         test_frac, random_state=0):
    """
    Pilot → train_seg + test_seg
    不同 segmentation algorithm 会使用不同 test_frac。
    例如：
        KMeans → test_frac = 0
        DAST → test_frac = 0.3
    """
    if test_frac <= 0:
        # No test split
        return (X_pilot, D_pilot, y_pilot), (None, None, None)

    X_tr, X_te, D_tr, D_te, y_tr, y_te = train_test_split(
        X_pilot, D_pilot, y_pilot,
        test_size=test_frac,
        random_state=random_state,
        stratify=D_pilot
    )
    
    # Convert to numpy arrays if they are pandas objects
    # This ensures consistent integer-based indexing (not label-based)
    # Note: After train_test_split, indices may be non-contiguous, so we reset first
    if hasattr(X_tr, 'reset_index'):
        X_tr = X_tr.reset_index(drop=True).values
        X_te = X_te.reset_index(drop=True).values
    elif hasattr(X_tr, 'values'):
        X_tr = X_tr.values
        X_te = X_te.values
    
    if hasattr(D_tr, 'reset_index'):
        D_tr = D_tr.reset_index(drop=True).values
        D_te = D_te.reset_index(drop=True).values
    elif hasattr(D_tr, 'values'):
        D_tr = D_tr.values
        D_te = D_te.values
    
    if hasattr(y_tr, 'reset_index'):
        y_tr = y_tr.reset_index(drop=True).values
        y_te = y_te.reset_index(drop=True).values
    elif hasattr(y_tr, 'values'):
        y_tr = y_tr.values
        y_te = y_te.values
    
    return (X_tr, D_tr, y_tr), (X_te, D_te, y_te)
