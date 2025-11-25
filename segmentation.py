# segmentation.py
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from outcome_model import predict_mu
from scoring import dams_score, kmeans_score
from dast import DASTree
from data_utils import  split_seg_train_test
from estimation import estimate_segment_policy  

class BaseSegmentation:
    """Base class for segmentation methods."""
    def fit(self, X, *args, **kwargs):
        raise NotImplementedError
    def assign(self, X):
        raise NotImplementedError


class KMeansSeg(BaseSegmentation):
    """K-Means based segmentation."""
    def __init__(self, n_segments, random_state=0):
        self.k = n_segments
        self.random_state = random_state
        self.model = None

    def fit(self, X, *args, **kwargs):
        self.model = KMeans(
            n_clusters=self.k,
            n_init=10,
            random_state=self.random_state
        ).fit(X)
        return self

    def assign(self, X):
        if self.model is None:
            raise RuntimeError("KMeansSeg: call fit() first")
        return self.model.predict(X)


# segmentation.py
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


class BaseSegmentation:
    """Base class for segmentation methods."""
    def fit(self, X, *args, **kwargs):
        raise NotImplementedError

    def assign(self, X):
        raise NotImplementedError


class KMeansSeg(BaseSegmentation):
    """K-Means based segmentation."""
    def __init__(self, n_segments, random_state=0):
        self.k = n_segments
        self.random_state = random_state
        self.model = None

    def fit(self, X, *args, **kwargs):
        self.model = KMeans(
            n_clusters=self.k,
            n_init=10,
            random_state=self.random_state
        ).fit(X)
        return self

    def assign(self, X):
        if self.model is None:
            raise RuntimeError("KMeansSeg: call fit() first")
        return self.model.predict(X)


class GMMSeg(BaseSegmentation):
    """Gaussian Mixture Model based segmentation."""
    def __init__(self, n_segments, covariance_type="full", random_state=0):
        self.k = n_segments
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.model = None

    def fit(self, X, *args, **kwargs):
        self.model = GaussianMixture(
            n_components=self.k,
            covariance_type=self.covariance_type,
            random_state=self.random_state
        )
        self.model.fit(X)
        return self

    def assign(self, X):
        if self.model is None:
            raise RuntimeError("GMMSeg: call fit() first")
        return self.model.predict(X)

    def bic(self, X):
        """Convenience wrapper for BIC on given data."""
        if self.model is None:
            raise RuntimeError("GMMSeg: call fit() before bic()")
        return self.model.bic(X)





# =========================================================
# 2. KMeans segmentation + K 选择
# =========================================================
def run_kmeans_segmentation(X_pilot, D_pilot, y_pilot, K_candidates):
    print("\n" + "=" * 60)
    print("STEP 4a: KMeans - selecting optimal K")
    print("=" * 60)

    best_K = None
    best_score = -np.inf

    for K in K_candidates:
        seg = KMeansSeg(K)
        seg.fit(X_pilot)

        score = kmeans_score(seg_model=seg, X_pilot=X_pilot,
                             D_pilot=D_pilot, y_pilot=y_pilot)
        print(f"  KMeans K={K} score={score:.4f}")

        if score > best_score:
            best_score = score
            best_K = K

    print(f"\n✓ KMeans: selected K = {best_K} with score = {best_score:.4f}\n")

    final_seg = KMeansSeg(best_K)
    final_seg.fit(X_pilot)
    seg_labels_pilot = final_seg.assign(X_pilot)

    return final_seg, seg_labels_pilot, best_K


# =========================================================
# 3. GMM segmentation + BIC 选 K
# =========================================================
def run_gmm_segmentation(X_pilot, D_pilot, y_pilot, K_candidates):
    print("\n" + "=" * 60)
    print("STEP 4b: GMM - selecting optimal K via BIC")
    print("=" * 60)

    best_K = None
    best_bic = np.inf

    for K in K_candidates:
        seg = GMMSeg(K)
        seg.fit(X_pilot)

        bic = seg.model.bic(X_pilot)
        print(f"  GMM K={K} BIC={bic:.1f}")

        if bic < best_bic:
            best_bic = bic
            best_K = K

    print(f"\n✓ GMM: selected K = {best_K} with BIC = {best_bic:.1f}\n")

    final_seg = GMMSeg(best_K)
    final_seg.fit(X_pilot)
    seg_labels_pilot = final_seg.assign(X_pilot)

    return final_seg, seg_labels_pilot, best_K


# =========================================================
# 4. DAST + DAMS（M selection）
# =========================================================
def run_dast_dams(
    X_pilot,
    D_pilot,
    y_pilot,
    mu1_pilot_model,
    mu0_pilot_model,
    e_pilot,
    Gamma_pilot,
    train_frac=0.7,
    M_candidates=(2, 3, 4, 5, 6, 7),
    min_leaf_size=20,
):
    print("\n" + "=" * 60)
    print("STEP 5: DAST - selecting optimal M via DAMS")
    print("=" * 60)

    d = X_pilot.shape[1]

    # train / val split
    print(
        f"Splitting pilot into train ({train_frac:.0%}) and "
        f"validation ({1 - train_frac:.0%})..."
    )
    (X_train, D_train, y_train), (X_val, D_val, y_val) = split_seg_train_test(
        X_pilot, D_pilot, y_pilot, test_frac=1 - train_frac
    )
    print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")

    # Gamma on train
    print("Computing pseudo-outcomes on train split...")
    mu1_train = predict_mu(mu1_pilot_model, X_train)
    mu0_train = predict_mu(mu0_pilot_model, X_train)

    Gamma1_train = mu1_train + (D_train / e_pilot) * (y_train - mu1_train)
    Gamma0_train = mu0_train + ((1 - D_train) / (1 - e_pilot)) * (y_train - mu0_train)
    Gamma_train = np.vstack([Gamma0_train, Gamma1_train]).T

    # candidate thresholds
    print("Computing candidate thresholds...")
    d_full = X_pilot.shape[1]
    # Generate candidate thresholds (midpoints between unique values)
    H_full = {}
    for j in range(d_full):
        sorted_values = np.sort(np.unique(X_pilot[:, j]))
        if len(sorted_values) > 1:
            H_full[j] = (sorted_values[:-1] + sorted_values[1:]) / 2.0
        else:
            H_full[j] = sorted_values

    print(f"Candidate thresholds computed for {d} features.")

    best_M = None
    best_score = -np.inf
    best_action = None

    print(f"\nTesting M candidates: {list(M_candidates)}")
    for M in M_candidates:
        tree = DASTree(
            x=X_train,
            y=y_train,
            D=D_train,
            gamma=Gamma_train,
            candidate_thresholds=H_full,
            min_leaf_size=min_leaf_size,
            max_depth=max(2, int(np.ceil(np.log2(M))) + 1),
        )
        tree.build()
        tree.prune_to_M(M)

        # segment labels on train + segment-level policy (diff-in-means on y)
        labels_train = tree.assign(X_train)
        tau_hat_M, action_M = estimate_segment_policy(
            X_train, y_train, D_train, labels_train
        )

        # DAMS scoring on validation (dual)
        score_M = dams_score(
            seg_model=tree,
            X_val=X_val,
            D_val=D_val,
            y_val=y_val,
            mu1_model=mu1_pilot_model,
            mu0_model=mu0_pilot_model,
            e=e_pilot,
            action=action_M,
        )
        print(f"  DAST M={M} DAMS-score={score_M:.6f}")

        if score_M >= best_score:
            best_score = score_M
            best_M = M
            best_action = action_M

    print(f"\n✓ DAST: selected M = {best_M} with DAMS-score = {best_score:.6f}\n")

    # 用 full pilot 重新 fit
    print("\n" + "=" * 60)
    print("STEP 6: Fitting final DAST on full pilot")
    print("=" * 60)

    print("Recomputing candidate thresholds on full pilot data...")
    
    tree_final = DASTree(
        x=X_pilot,
        y=y_pilot,
        D=D_pilot,
        gamma=Gamma_pilot,
        candidate_thresholds=H_full,
        min_leaf_size=min_leaf_size,
        max_depth=max(2, int(np.ceil(np.log2(best_M))) + 1),
    )
    tree_final.build()
    tree_final.prune_to_M(best_M)
    seg_labels_pilot = tree_final.assign(X_pilot)

    return tree_final, seg_labels_pilot, best_M, best_action

