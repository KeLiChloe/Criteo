# data_utils.py
from sklearn.model_selection import train_test_split
from sklift.datasets import fetch_criteo
from outcome_model import fit_mu_models, predict_mu
import numpy as np
from sklearn.preprocessing import StandardScaler

def split_pilot_impl(X, D, y, pilot_frac, random_state=0):
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


def split_seg_train_test(X_pilot, D_pilot, y_pilot, Gamma_pilot,
                         test_frac):
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

    X_tr, X_te, D_tr, D_te, y_tr, y_te, Gamma_tr, Gamma_te = train_test_split(
        X_pilot, D_pilot, y_pilot, Gamma_pilot,
        test_size=test_frac,
        random_state=0,
        stratify=y_pilot)

    
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
    
    return (X_tr, D_tr, y_tr, Gamma_tr), (X_te, D_te, y_te, Gamma_te)



# =========================================================
# 0. 数据加载 & 探索
# =========================================================
def load_criteo(sample_frac, seed):
    np.random.seed(seed)
    print("Loading Criteo uplift dataset ...")
    print("(Using random seed =", seed, ")")
    
    X, y, D = fetch_criteo(
        target_col="conversion",
        treatment_col="treatment",
        percent10=True,
        return_X_y_t=True,
    )
    

    n_samples = int(len(X) * sample_frac)
    indices = np.random.choice(len(X), size=n_samples, replace=False)
    X, y, D = X.iloc[indices], y.iloc[indices], D.iloc[indices]
    
    # print posit=iive ratio of y
    print(f"Positive ratio of y: {y.mean():.6f}")

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    D = D.reset_index(drop=True)

    print("\n" + "=" * 60)
    print("DATA EXPLORATION")
    print("=" * 60)

    print("\n Basic Information:")
    print(f"   X shape: {X.shape} (n={X.shape[0]}, d={X.shape[1]})")

    # print("\n Outcome by Treatment:")
    # y_control = y[D == 0]
    # y_treated = y[D == 1]
    # print(f"   Control (D=0) - mean: {y_control.mean():.6f}, std: {y_control.std():.6f}")
    # print(f"   Treated (D=1) - mean: {y_treated.mean():.6f}, std: {y_treated.std():.6f}")
    # print(f"   Naive ATE: {y_treated.mean() - y_control.mean():.6f}")
    
    # # print ratio of treatment assignment (D=1) and positive outcomes
    # print("\n Treatment Assignment:")
    # print(f"   Treatment (D=1) ratio: {D.mean():.6f}")
    # print(f"   Positive Outcome (y=1) ratio: {y.mean():.6f}")

    # 转成 numpy
    X_np = X.values
    y_np = y.values
    D_np = D.values

    # scale X features
    # scaler = StandardScaler()
    # X_np = scaler.fit_transform(X_np)
    
    return X_np, y_np, D_np


# =========================================================
# 1. pilot / implementation 划分 + outcome model + Gamma
# =========================================================
def prepare_pilot_impl(X, y, D, pilot_frac):
    print("\n" + "=" * 60)
    print("Split & fit outcome models")
    print("=" * 60)

    X_pilot, X_impl, D_pilot, D_impl, y_pilot, y_impl = split_pilot_impl(
        X, D, y, pilot_frac=pilot_frac
    )
    print(f"Pilot size: {len(X_pilot)}, Implementation size: {len(X_impl)}")

    mu1_pilot_model, mu0_pilot_model = fit_mu_models(
        X_pilot, D_pilot, y_pilot, model_type="logistic"
    )
    e_pilot = D_pilot.mean()

    mu1_pilot = predict_mu(mu1_pilot_model, X_pilot)
    mu0_pilot = predict_mu(mu0_pilot_model, X_pilot)

    Gamma1_pilot = mu1_pilot + (D_pilot / e_pilot) * (y_pilot - mu1_pilot)
    Gamma0_pilot = mu0_pilot + ((1 - D_pilot) / (1 - e_pilot)) * (y_pilot - mu0_pilot)
    Gamma_pilot = np.vstack([Gamma0_pilot, Gamma1_pilot]).T

    return (
        X_pilot,
        X_impl,
        D_pilot,
        D_impl,
        y_pilot,
        y_impl,
        mu1_pilot_model,
        mu0_pilot_model,
        e_pilot,
        Gamma_pilot,
    )

