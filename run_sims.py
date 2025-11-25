# exp_sim.py
"""
多次重复 Criteo 实验，并保存每次各算法的 value_mean 到 pkl。

- 每一轮：
    1) 用一个新的 random_state 调用 load_criteo(sample_frac=0.03, random_state=seed)
    2) prepare_pilot_impl → 得到 pilot / impl, mu1/mu0, Gamma_pilot, e_pilot
    3) 跑 KMeans / GMM / DAST（用你已有的 run_xxx_segmentation, run_dast_dams）
    4) 用 estimate_segment_policy + evaluate_policy 计算：
        - Baseline: All Treat
        - Baseline: All Control
        - Baseline: Random (按 e_pilot 随机 treat)
        - KMeans policy
        - GMM policy
        - DAST policy
    5) 把本轮结果（一个 dict） append 到列表，并立刻 pickle.dump 覆盖保存

结果文件：比如 "criteo_sim_results.pkl"
"""

import numpy as np
import pickle
import os

from data_utils import load_criteo, prepare_pilot_impl
from estimation import estimate_segment_policy
from evaluation import evaluate_policy
from segmentation import (
    run_kmeans_segmentation,
    run_gmm_segmentation,
    run_dast_dams,
)


def run_single_experiment(sample_frac=0.03, pilot_frac=0.4, seed=None):
    """
    跑一轮完整实验，返回一个 dict，里面放所有算法的 value_mean。
    """

    if seed is None:
        seed = np.random.randint(0, 1_000_000)
    print("\n" + "=" * 60)
    print(f"▶ Running one experiment with seed = {seed}")
    print("=" * 60)

    # --------------------------------------------------
    # 0. Load Criteo 子样本
    # --------------------------------------------------
    X, y, D = load_criteo(sample_frac=sample_frac, random_state=seed)

    # --------------------------------------------------
    # 1–3. pilot / impl + outcome models + Gamma
    # --------------------------------------------------
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
        Gamma_pilot,
    ) = prepare_pilot_impl(X, y, D, pilot_frac=pilot_frac)

    # --------------------------------------------------
    # 4a. KMeans segmentation
    # --------------------------------------------------
    kmeans_seg, seg_labels_pilot_kmeans, best_K_kmeans = run_kmeans_segmentation(
        X_pilot, D_pilot, y_pilot, K_candidates=[2, 3, 4]
    )

    # --------------------------------------------------
    # 4b. GMM segmentation
    # --------------------------------------------------
    gmm_seg, seg_labels_pilot_gmm, best_K_gmm = run_gmm_segmentation(
        X_pilot, D_pilot, y_pilot, K_candidates=[2, 3, 4, 5, 6]
    )

    # --------------------------------------------------
    # 5–6. DAST + DAMS
    # --------------------------------------------------
    (
        tree_final,
        seg_labels_pilot_dast,
        best_M_dast,
        best_action_dast_train,  # just for sanity check
    ) = run_dast_dams(
        X_pilot,
        D_pilot,
        y_pilot,
        mu1_pilot_model,
        mu0_pilot_model,
        e_pilot,
        Gamma_pilot,
        train_frac=0.5,
        M_candidates=(2, 3, 4, 5, 6),
        min_leaf_size=5,
    )

    # --------------------------------------------------
    # 7. 估计 segment-level policy（diff-in-means on y）
    # --------------------------------------------------
    tau_kmeans, action_kmeans = estimate_segment_policy(
        X_pilot, y_pilot, D_pilot, seg_labels_pilot_kmeans
    )
    tau_gmm, action_gmm = estimate_segment_policy(
        X_pilot, y_pilot, D_pilot, seg_labels_pilot_gmm
    )
    tau_dast, action_dast = estimate_segment_policy(
        X_pilot, y_pilot, D_pilot, seg_labels_pilot_dast
    )

    # --------------------------------------------------
    # 8. Implementation assignment
    # --------------------------------------------------
    seg_labels_impl_kmeans = kmeans_seg.assign(X_impl)
    seg_labels_impl_gmm = gmm_seg.assign(X_impl)
    seg_labels_impl_dast = tree_final.assign(X_impl)

    # --------------------------------------------------
    # 9. Baselines + algorithm policies: evaluate on impl
    # --------------------------------------------------
    from evaluation import evaluate_policy  # already imported above, just to be clear

    # Baseline: All Treat
    seg_labels_all_treat = np.zeros(len(X_impl), dtype=int)  # all in segment 0
    action_all_treat = np.array([1])  # segment 0 → treat
    value_all_treat = evaluate_policy(
        X_impl,
        D_impl,
        y_impl,
        seg_labels_all_treat,
        mu1_pilot_model,
        mu0_pilot_model,
        action_all_treat,
        e_pilot,
    )

    # Baseline: All Control
    seg_labels_all_control = np.zeros(len(X_impl), dtype=int)
    action_all_control = np.array([0])
    value_all_control = evaluate_policy(
        X_impl,
        D_impl,
        y_impl,
        seg_labels_all_control,
        mu1_pilot_model,
        mu0_pilot_model,
        action_all_control,
        e_pilot,
    )

    # Baseline: Random (按 e_pilot 概率 treat)
    # 做法：segment=0 表示控制段，segment=1 表示处理段；action=[0,1]
    seg_labels_random = np.random.binomial(1, e_pilot, size=len(X_impl))
    action_random = np.array([0, 1])
    value_random = evaluate_policy(
        X_impl,
        D_impl,
        y_impl,
        seg_labels_random,
        mu1_pilot_model,
        mu0_pilot_model,
        action_random,
        e_pilot,
    )

    # KMeans policy
    value_kmeans = evaluate_policy(
        X_impl,
        D_impl,
        y_impl,
        seg_labels_impl_kmeans,
        mu1_pilot_model,
        mu0_pilot_model,
        action_kmeans,
        e_pilot,
    )

    # GMM policy
    value_gmm = evaluate_policy(
        X_impl,
        D_impl,
        y_impl,
        seg_labels_impl_gmm,
        mu1_pilot_model,
        mu0_pilot_model,
        action_gmm,
        e_pilot,
    )

    # DAST policy
    value_dast = evaluate_policy(
        X_impl,
        D_impl,
        y_impl,
        seg_labels_impl_dast,
        mu1_pilot_model,
        mu0_pilot_model,
        action_dast,
        e_pilot,
    )

    # --------------------------------------------------
    # 打个 summary，顺便返回 v_mean
    # --------------------------------------------------
    res = {
        "seed": int(seed),
        "e_pilot": float(e_pilot),
        "all_treat": float(value_all_treat["value_mean"]),
        "all_control": float(value_all_control["value_mean"]),
        "random": float(value_random["value_mean"]),
        "kmeans": float(value_kmeans["value_mean"]),
        "gmm": float(value_gmm["value_mean"]),
        "dast": float(value_dast["value_mean"]),
    }

    print("\nResult for this run (value_mean):")
    for k, v in res.items():
        if k in ["seed", "e_pilot"]:
            print(f"  {k}: {v}")
        else:
            print(f"  {k:12s}: {v:.6f}")

    return res


def run_multiple_experiments(
    N_sim=10,
    sample_frac=0.03,
    pilot_frac=0.4,
    out_path="criteo_sim_results.pkl",
):
    """
    重复跑 N_sim 次实验；每跑完一轮就覆盖保存一次 pkl。
    """
    all_results = []

    print("\n" + "=" * 60)
    print(f"STARTING SIMULATIONS: N_sim = {N_sim}")
    print("=" * 60)

    for s in range(N_sim):
        # 每一轮用一个新的 seed
        seed = np.random.randint(0, 1_000_000)
        res = run_single_experiment(
            sample_frac=sample_frac,
            pilot_frac=pilot_frac,
            seed=seed,
        )
        all_results.append(res)

        # 每轮都覆盖保存一次
        with open(out_path, "wb") as f:
            pickle.dump(all_results, f)

        print(
            f"\n[SIM {s+1}/{N_sim}] saved {len(all_results)} runs to '{out_path}'"
        )
        print("-" * 60)

    print("\n" + "=" * 60)
    print("ALL SIMULATIONS DONE.")
    print(f"Results saved in '{out_path}'")
    print("=" * 60)


if __name__ == "__main__":
    # 你可以在这里改 N_sim 或输出路径
    run_multiple_experiments(
        N_sim=20,
        sample_frac=0.02,
        pilot_frac=0.4,
        out_path="criteo_sim_results.pkl",
    )
