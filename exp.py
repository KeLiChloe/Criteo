# exp.py  —— Criteo 上对比 KMeans / GMM / DAST 的完整实验脚本

import numpy as np
from sklift.datasets import fetch_criteo

from data_utils import split_pilot_impl, split_seg_train_test
from outcome_model import fit_mu_models, predict_mu
from estimation import estimate_segment_policy    # 版本: estimate_segment_policy(X, y, D, seg_labels)
from evaluation import evaluate_policy_dr
from segmentation import KMeansSeg, GMMSeg       # 需要在 segmentation.py 里实现 GMMSeg
from scoring import dams_score, kmeans_score
from dast import DASTree


# =========================================================
# 0. 数据加载 & 探索
# =========================================================
def load_criteo(sample_frac=0.05, random_state=None):
    print("Loading Criteo uplift dataset ...")
    X, y, D = fetch_criteo(
        target_col="visit",
        treatment_col="treatment",
        percent10=True,
        return_X_y_t=True,
    )

    if random_state is None:
        random_state = np.random.randint(0, 1_000_000)
    np.random.seed(random_state)

    n_samples = int(len(X) * sample_frac)
    indices = np.random.choice(len(X), size=n_samples, replace=False)
    X, y, D = X.iloc[indices], y.iloc[indices], D.iloc[indices]

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    D = D.reset_index(drop=True)

    print("\n" + "=" * 60)
    print("DATA EXPLORATION")
    print("=" * 60)

    print("\n Basic Information:")
    print(f"   X shape: {X.shape} (n={X.shape[0]}, d={X.shape[1]})")

    print("\n Outcome by Treatment:")
    y_control = y[D == 0]
    y_treated = y[D == 1]
    print(f"   Control (D=0) - mean: {y_control.mean():.6f}, std: {y_control.std():.6f}")
    print(f"   Treated (D=1) - mean: {y_treated.mean():.6f}, std: {y_treated.std():.6f}")
    print(f"   Naive ATE: {y_treated.mean() - y_control.mean():.6f}")

    # 转成 numpy
    X_np = X.values
    y_np = y.values
    D_np = D.values

    return X_np, y_np, D_np


# =========================================================
# 1. pilot / implementation 划分 + outcome model + Gamma
# =========================================================
def prepare_pilot_impl(X, y, D, pilot_frac=0.3):
    print("\n" + "=" * 60)
    print("STEP 1–3: Split & fit outcome models")
    print("=" * 60)

    X_pilot, X_impl, D_pilot, D_impl, y_pilot, y_impl = split_pilot_impl(
        X, D, y, pilot_frac=pilot_frac
    )
    print(f"Pilot size: {len(X_pilot)}, Implementation size: {len(X_impl)}")

    print("\nFitting outcome models (mu1, mu0) on pilot...")
    mu1_pilot_model, mu0_pilot_model = fit_mu_models(
        X_pilot, D_pilot, y_pilot, model_type="logistic"
    )
    e_pilot = D_pilot.mean()
    print(f"Propensity score e: {e_pilot:.3f}")

    print("Computing pseudo-outcomes (Gamma) on pilot data...")
    mu1_pilot = predict_mu(mu1_pilot_model, X_pilot)
    mu0_pilot = predict_mu(mu0_pilot_model, X_pilot)

    Gamma1_pilot = mu1_pilot + (D_pilot / e_pilot) * (y_pilot - mu1_pilot)
    Gamma0_pilot = mu0_pilot + ((1 - D_pilot) / (1 - e_pilot)) * (y_pilot - mu0_pilot)
    Gamma_pilot = np.vstack([Gamma0_pilot, Gamma1_pilot]).T
    print("Pseudo-outcomes computed.")

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
        Gamma1_pilot,
        Gamma0_pilot,
        Gamma_pilot,
    )


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
    H = [np.quantile(X_train[:, j], np.linspace(0.1, 0.9, 5)) for j in range(d)]
    print(f"Candidate thresholds computed for {d} features.")

    best_M = None
    best_score = -np.inf
    best_action = None

    print(f"\nTesting M candidates: {list(M_candidates)}")
    for M in M_candidates:
        print(f"\n  Building DAST with M={M}...")

        tree = DASTree(
            x=X_train,
            y=y_train,
            D=D_train,
            gamma=Gamma_train,
            candidate_thresholds=H,
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
    d_full = X_pilot.shape[1]
    H_full = [np.quantile(X_pilot[:, j], np.linspace(0.1, 0.9, 5)) for j in range(d_full)]

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


# =========================================================
# 5. 评估所有 policy（含 normalized & relative）
# =========================================================
def evaluate_all_policies(
    X_impl,
    y_impl,
    D_impl,
    seg_labels_impl_kmeans,
    seg_labels_impl_gmm,
    seg_labels_impl_dast,
    mu1_pilot_model,
    mu0_pilot_model,
    e_pilot,
    action_kmeans,
    action_gmm,
    action_dast,
):
    print("\n" + "=" * 60)
    print("STEP 9: Evaluating policies on implementation data")
    print("=" * 60)

    # baseline: all treat
    print("\nEvaluating baseline: All Treat...")
    seg_all_treat = np.zeros(len(X_impl), dtype=int)
    action_all_treat = np.array([1])
    val_all_treat = evaluate_policy_dr(
        X_impl,
        D_impl,
        y_impl,
        seg_all_treat,
        mu1_pilot_model,
        mu0_pilot_model,
        action_all_treat,
        e_pilot,
    )

    # baseline: all control
    print("Evaluating baseline: All Control...")
    seg_all_control = np.zeros(len(X_impl), dtype=int)
    action_all_control = np.array([0])
    val_all_control = evaluate_policy_dr(
        X_impl,
        D_impl,
        y_impl,
        seg_all_control,
        mu1_pilot_model,
        mu0_pilot_model,
        action_all_control,
        e_pilot,
    )

    # KMeans
    print("Evaluating KMeans policy...")
    val_kmeans = evaluate_policy_dr(
        X_impl,
        D_impl,
        y_impl,
        seg_labels_impl_kmeans,
        mu1_pilot_model,
        mu0_pilot_model,
        action_kmeans,
        e_pilot,
    )

    # GMM
    print("Evaluating GMM policy...")
    val_gmm = evaluate_policy_dr(
        X_impl,
        D_impl,
        y_impl,
        seg_labels_impl_gmm,
        mu1_pilot_model,
        mu0_pilot_model,
        action_gmm,
        e_pilot,
    )

    # DAST
    print("Evaluating DAST policy...")
    val_dast = evaluate_policy_dr(
        X_impl,
        D_impl,
        y_impl,
        seg_labels_impl_dast,
        mu1_pilot_model,
        mu0_pilot_model,
        action_dast,
        e_pilot,
    )

    # ================= Metrics: absolute / normalized / relative =================
    v_ctrl = val_all_control["value_mean"]
    v_treat = val_all_treat["value_mean"]
    denom = max(v_treat - v_ctrl, 1e-12)  # avoid divide-by-zero

    methods = {
        "all_treat": val_all_treat,
        "all_control": val_all_control,
        "kmeans": val_kmeans,
        "gmm": val_gmm,
        "dast": val_dast,
    }

    print("\n" + "=" * 60)
    print("RESULTS (absolute)")
    print("=" * 60)
    for name, val in methods.items():
        vm = val["value_mean"]
        print(
            f"{name:12s}  value_mean = {vm: .6f}   "
        )

    # =====================================================
    # 相对提升：DAST vs 每个 baseline
    # =====================================================
    print("\n" + "=" * 60)
    print("RELATIVE IMPROVEMENT")
    print("=" * 60)

    baselines = {
        "All Control": val_all_control,
        "All Treat": val_all_treat,
        "KMeans": val_kmeans,
        "GMM": val_gmm,
    }

    v_dast = val_dast["value_mean"]

    # 用于计算相对提升的分母
    def safe_div(a, b):
        return a / (b + 1e-12)

    for name, val in baselines.items():
        v_base = val["value_mean"]
        abs_gain = v_dast - v_base
        rel_gain = safe_div(abs_gain, abs(v_base)) * 100   # 相对提升 %

        print(f"{name:12s}:  "
              f"abs Δ = {abs_gain:+.6f},   "
              f"relative = {rel_gain:+.2f}%")

    print("=" * 60)



# =========================================================
# main
# =========================================================
def main():
    # 0. load data
    X, y, D = load_criteo(sample_frac=0.05)

    # 1–3. pilot / impl + outcome models + Gamma
    (
        X_pilot,
        X_impl,
        D_pilot,
        D_impl,
        y_pilot,
        y_impl,
        mu1_pilot_model,
        mu0_pilot_model,
        e_pilot,
        Gamma1_pilot,
        Gamma0_pilot,
        Gamma_pilot,
    ) = prepare_pilot_impl(X, y, D, pilot_frac=0.3)

    # 4a. KMeans
    kmeans_seg, seg_labels_pilot_kmeans, best_K_kmeans = run_kmeans_segmentation(
        X_pilot, D_pilot, y_pilot, K_candidates=[2, 3, 4, 5, 6]
    )

    # 4b. GMM
    gmm_seg, seg_labels_pilot_gmm, best_K_gmm = run_gmm_segmentation(
        X_pilot, D_pilot, y_pilot, K_candidates=[2, 3, 4, 5, 6]
    )

    # 5–6. DAST + DAMS
    (
        tree_final,
        seg_labels_pilot_dast,
        best_M_dast,
        best_action_dast_train,  # 可以用来 sanity check
    ) = run_dast_dams(
        X_pilot,
        D_pilot,
        y_pilot,
        mu1_pilot_model,
        mu0_pilot_model,
        e_pilot,
        Gamma_pilot,
        train_frac=0.7,
        M_candidates=(2, 3, 4, 5, 6),
        min_leaf_size=20,
    )

    # 7. 估计 segment-level policy（diff-in-means on y）
    print("\n" + "=" * 60)
    print("STEP 7: Estimating segment policies")
    print("=" * 60)

    print("Estimating KMeans policy...")
    tau_kmeans, action_kmeans = estimate_segment_policy(
        X_pilot, y_pilot, D_pilot, seg_labels_pilot_kmeans
    )
    print(
        f"KMeans - Segments: {len(np.unique(seg_labels_pilot_kmeans))}, "
        f"Actions: {action_kmeans}"
    )

    print("Estimating GMM policy...")
    tau_gmm, action_gmm = estimate_segment_policy(
        X_pilot, y_pilot, D_pilot, seg_labels_pilot_gmm
    )
    print(
        f"GMM    - Segments: {len(np.unique(seg_labels_pilot_gmm))}, "
        f"Actions: {action_gmm}"
    )

    print("Estimating DAST policy...")
    tau_dast, action_dast = estimate_segment_policy(
        X_pilot, y_pilot, D_pilot, seg_labels_pilot_dast
    )
    print(
        f"DAST   - Segments: {len(np.unique(seg_labels_pilot_dast))}, "
        f"Actions: {action_dast}"
    )

    # 8. implementation segment assignment
    print("\n" + "=" * 60)
    print("STEP 8: Assigning implementation data to segments")
    print("=" * 60)
    seg_labels_impl_kmeans = kmeans_seg.assign(X_impl)
    seg_labels_impl_gmm = gmm_seg.assign(X_impl)
    seg_labels_impl_dast = tree_final.assign(X_impl)
    print("Implementation assignments complete.")

    # 9. evaluation
    evaluate_all_policies(
        X_impl,
        y_impl,
        D_impl,
        seg_labels_impl_kmeans,
        seg_labels_impl_gmm,
        seg_labels_impl_dast,
        mu1_pilot_model,
        mu0_pilot_model,
        e_pilot,
        action_kmeans,
        action_gmm,
        action_dast,
    )


if __name__ == "__main__":
    main()
