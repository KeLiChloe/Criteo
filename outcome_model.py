import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# LightGBM
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


# =====================================================================
#  Utility: safe fit (fallback when only 1 class is present)
# =====================================================================


def safe_fit(model, X, y):
    model.fit(X, y)
    return model


# =====================================================================
#  Fit μ₁(x), μ₀(x) under different model types
# =====================================================================

def fit_mu_models(X, D, y, model_type):
    """
    Fit outcome models:
        μ1(x) = E[Y | X, D=1]
        μ0(x) = E[Y | X, D=0]

    model_type ∈ {"logistic", "lightgbm", "mlp"}
    """

    mask_t = (D == 1)
    mask_c = (D == 0)

    # --------------------------
    # 1. Create model by type
    # --------------------------
    def make_model():
        if model_type == "logistic":
            return LogisticRegression(max_iter=300)

        elif model_type == "mlp":
            return MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                max_iter=300,
            )

        elif model_type == "lightgbm":
            if not HAS_LGBM:
                raise ImportError("LightGBM not installed. Run: pip install lightgbm")
            return LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=-1,
                class_weight='balanced',
            )

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    # --------------------------
    # 2. Fit μ1, μ0
    # --------------------------
    model1 = make_model()
    model0 = make_model()

    mu1 = safe_fit(model1, X[mask_t], y[mask_t])
    mu0 = safe_fit(model0, X[mask_c], y[mask_c])

    return mu1, mu0


# =====================================================================
#  Predict μ(x)
# =====================================================================

def predict_mu(mu_model, X):
    """Return predicted probability P(y=1 | x)."""
    return mu_model.predict_proba(X)[:, 1]
