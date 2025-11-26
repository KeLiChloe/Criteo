# segmentation.py
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from outcome_model import predict_mu
from scoring import dams_score, kmeans_score
from dast import DASTree
from data_utils import  split_seg_train_test
from estimation import estimate_segment_policy  
from clr import CLRSeg, clr_bic_score
from policytree import PolicyTreeSeg, _fit_policytree_with_grf

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

    bins = 200
    H_full = {}

    for j in range(d_full):
        col = X_pilot[:, j]
        # 去掉nan的话可以先 col = col[~np.isnan(col)]
        unique_values = np.unique(col)


        # 如果 unique 太多，只取 K+1 个“代表点”，再在中间算 midpoints
        if len(unique_values) > bins + 1:
            # 取 K+1 个分位数，比如 [0, 1/K, 2/K, ..., 1]
            qs = np.linspace(0, 1, num=bins+1)
            # 用 quantile 近似 unique-values 的分布
            grid = np.quantile(col, qs)
            grid = np.unique(grid)  # 可能有重复
        else:
            grid = unique_values

        if len(grid) > 1:
            H_full[j] = (grid[:-1] + grid[1:]) / 2.0
        else:
            H_full[j] = grid


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



from clr import CLRSeg, clr_bic_score

def run_clr_segmentation(
    X_pilot,
    D_pilot,
    y_pilot,
    K_candidates=(2, 3, 4, 5),
    kmeans_coef=0.1,
    num_tries=8,
    random_state=0,
):
    best_K = None
    best_score = np.inf
    best_seg = None
    best_labels = None

    for K in K_candidates:
        seg = CLRSeg(
            n_segments=K,
            kmeans_coef=kmeans_coef,
            num_tries=num_tries,
            random_state=random_state,
        )
        seg.fit(X_pilot, D_pilot, y_pilot)
        bic = clr_bic_score(seg, X_pilot, D_pilot, y_pilot)
        print(f"CLR K={K} BIC={bic:.3f}")

        if bic < best_score and bic > -np.inf:
            best_score = bic
            best_K = K
            best_seg = seg
            best_labels = seg.assign(X_pilot)

    print(f"\n✓ CLR selected K={best_K} with BIC={best_score:.3f}\n")
    return best_seg, best_labels, best_K




# segmentation.py 里，放在 run_dast_dams 附近
import numpy as np
from data_utils import split_seg_train_test
from outcome_model import predict_mu
from estimation import estimate_segment_policy
from scoring import dams_score
from mst import MSTree   # ← 别忘了从你新建的 mst.py 引入


def run_mst_dams(
    X_pilot,
    D_pilot,
    y_pilot,
    mu1_pilot_model,
    mu0_pilot_model,
    e_pilot,
    train_frac=0.7,
    M_candidates=(2, 3, 4, 5, 6, 7),
    min_leaf_size=20,
):
    print("\n" + "=" * 60)
    print("STEP 5 (MST): selecting optimal M via DAMS (residual-based splits)")
    print("=" * 60)

    d = X_pilot.shape[1]

    # --------------------------------------------------
    # train / val split
    # --------------------------------------------------
    print(
        f"Splitting pilot into train ({train_frac:.0%}) and "
        f"validation ({1 - train_frac:.0%})..."
    )
    (X_train, D_train, y_train), (X_val, D_val, y_val) = split_seg_train_test(
        X_pilot, D_pilot, y_pilot, test_frac=1 - train_frac
    )
    print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")

    # --------------------------------------------------
    # candidate thresholds: 跟 run_dast_dams 一样
    # --------------------------------------------------
    print("Computing candidate thresholds (MST)...")
    d_full = X_pilot.shape[1]
    bins = 20
    H_full = {}

    for j in range(d_full):
        col = X_pilot[:, j]
        unique_values = np.unique(col)

        if len(unique_values) <= 1:
            H_full[j] = unique_values
        else:
            if len(unique_values) > bins + 1:
                qs = np.linspace(0, 1, num=bins + 1)
                grid = np.quantile(col, qs)
                grid = np.unique(grid)
            else:
                grid = unique_values

            if len(grid) > 1:
                H_full[j] = (grid[:-1] + grid[1:]) / 2.0
            else:
                H_full[j] = grid

    print(f"Candidate thresholds computed for {d} features.")

    # --------------------------------------------------
    # DAMS: loop over M
    # --------------------------------------------------
    best_M = None
    best_score = -np.inf
    best_action = None

    print(f"\nTesting M candidates for MST: {list(M_candidates)}")
    for M in M_candidates:
        tree = MSTree(
            x=X_train,
            y=y_train,
            D=D_train,
            candidate_thresholds=H_full,
            min_leaf_size=min_leaf_size,
            max_depth=max(2, int(np.ceil(np.log2(M))) + 1),
            epsilon=0.0,
        )
        tree.build()
        tree.prune_to_M(M)

        # segment labels on train + segment-level policy (diff-in-means on y)
        labels_train = tree.assign(X_train)
        tau_hat_M, action_M = estimate_segment_policy(
            X_train, y_train, D_train, labels_train
        )

        # DAMS scoring on validation (dual) —— 跟 DAST 完全一样
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
        print(f"  MST  M={M} DAMS-score={score_M:.6f}")

        if score_M >= best_score:
            best_score = score_M
            best_M = M
            best_action = action_M

    print(f"\n✓ MST: selected M = {best_M} with DAMS-score = {best_score:.6f}\n")

    # --------------------------------------------------
    # 用 full pilot 重新 fit
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 6 (MST): Fitting final MST on full pilot")
    print("=" * 60)

    print("Reusing candidate thresholds H_full on full pilot...")

    tree_final = MSTree(
        x=X_pilot,
        y=y_pilot,
        D=D_pilot,
        candidate_thresholds=H_full,
        min_leaf_size=min_leaf_size,
        max_depth=max(2, int(np.ceil(np.log2(best_M))) + 1),
        epsilon=0.0,
    )
    tree_final.build()
    tree_final.prune_to_M(best_M)
    seg_labels_pilot = tree_final.assign(X_pilot)

    return tree_final, seg_labels_pilot, best_M, best_action





# =====================================================================
#  外部调用：run_policytree_segmentation（带 DAMS 选 M）
# =====================================================================

def run_policytree_segmentation(
    X_pilot: np.ndarray,
    D_pilot: np.ndarray,
    y_pilot: np.ndarray,
    mu1_pilot_model,
    mu0_pilot_model,
    e_pilot: float,
    depth: int = 2,
    train_frac: float = 0.7,
    M_candidates=(2, 3, 4, 5),
    min_leaf_size: int = 20,
):
    """
    和 run_dast_dams 类似的接口：

      输入：
        - X_pilot, D_pilot, y_pilot：pilot 数据
        - mu1_pilot_model, mu0_pilot_model, e_pilot：用于 DAMS 的 DR
        - depth：policy_tree 的最大深度
        - train_frac：train / val 划分
        - M_candidates：候选 segment 个数
      输出：
        - seg_model_final: PolicyTreeSeg（在 full pilot 上重训 + prune）
        - seg_labels_pilot: full pilot 上的 segment_id
        - best_M: 选出来的 M
        - best_action: 在 full pilot 上重估的 segment 行为（diff-in-means 学的）
    """

    print("\n" + "=" * 60)
    print("POLICYTREE: selecting M via DAMS (Gamma from R)")
    print("=" * 60)

    # 1) train/val split
    (X_train, D_train, y_train), (X_val, D_val, y_val) = split_seg_train_test(
        X_pilot, D_pilot, y_pilot,
        test_frac=1 - train_frac
    )

    print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")
    print(f"Fitting GRF + PolicyTree on train with depth={depth} ...")

    # 2) 在 train 上用 R 拟合 policy_tree + Gamma
    tree_r_train, Gamma_train, leaf_parent_map, leaf_ids_train, action_ids_train = \
        _fit_policytree_with_grf(X_train, y_train, D_train, depth=depth)

    # 换算 leaf 数量的上限
    n_initial_segments = len(np.unique(leaf_ids_train))
    M_candidates = [M for M in M_candidates if M <= n_initial_segments]
    if not M_candidates:
        raise ValueError(f"No valid M_candidates <= initial leaf count {n_initial_segments}")

    print(f"Initial leaf count (train): {n_initial_segments}")
    print(f"Testing M candidates: {M_candidates}")

    best_M = None
    best_score = -np.inf
    best_action = None
    best_leaf_to_pruned = None

    # 3) 对每个 M 做 post-pruning + DAMS
    for M in M_candidates:
        print(f"\n  >> M = {M}")
        seg_labels_train, action_ids_seg, leaf_to_pruned = policytree_post_prune_tree(
            leaf_ids_train,
            action_ids_train,
            Gamma_train,
            target_leaf_num=M,
            leaf_to_parent_map=leaf_parent_map
        )

        # 基于 train segmentation，用 diff-in-means(y) 学各 segment 的 action
        tau_hat_M, action_M = estimate_segment_policy(
            X_train, y_train, D_train, seg_labels_train
        )

        # 构造一个临时 segmentation model（只用于 DAMS scoring）
        seg_model_tmp = PolicyTreeSeg(tree_r_train, leaf_to_pruned)

        score_M = dams_score(
            seg_model=seg_model_tmp,
            X_val=X_val,
            D_val=D_val,
            y_val=y_val,
            mu1_model=mu1_pilot_model,
            mu0_model=mu0_pilot_model,
            e=e_pilot,
            action=action_M,
        )
        print(f"    DAMS-score = {score_M:.6f}")

        if score_M >= best_score:
            best_score = score_M
            best_M = M
            best_action = action_M
            best_leaf_to_pruned = leaf_to_pruned

    print(f"\n✓ POLICYTREE: selected M = {best_M} with DAMS-score = {best_score:.6f}")

    # 4) 在 full pilot 上重训一次 policy_tree + Gamma，再 prune 到 best_M

    print("\nRe-fitting GRF + PolicyTree on FULL pilot ...")
    tree_r_full, Gamma_full, leaf_parent_full, leaf_ids_full, action_ids_full = \
        _fit_policytree_with_grf(X_pilot, y_pilot, D_pilot, depth=depth)

    # 注意：full pilot 上的 initial leaf 数可能和 train 不同，
    # 这里再 prune 一次到 best_M （如果不够就用实际 leaf 数）
    n_leaves_full = len(np.unique(leaf_ids_full))
    target_M_full = min(best_M, n_leaves_full)

    seg_labels_full, action_ids_full_seg, leaf_to_pruned_full = policytree_post_prune_tree(
        leaf_ids_full,
        action_ids_full,
        Gamma_full,
        target_leaf_num=target_M_full,
        leaf_to_parent_map=leaf_parent_full,
    )

    # full pilot 上再用 diff-in-means(y) 学一遍 action
    tau_final, action_final = estimate_segment_policy(
        X_pilot, y_pilot, D_pilot, seg_labels_full
    )

    seg_model_final = PolicyTreeSeg(tree_r_full, leaf_to_pruned_full)

    return seg_model_final, seg_labels_full, target_M_full, action_final
