# exp_sim.py
"""
多次重复 Criteo 实验，并保存每次各算法（包括 CLR）的 value_mean 到 pkl。
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
    run_clr_segmentation,   
    run_mst_dams,    
    run_policytree_segmentation,
)

import time  

ALGO_LIST = [ "policytree", "dast", "kmeans", "gmm", "clr", "dast", "mst"] # ["kmeans", "gmm", "clr", "dast", "mst"]  # 可选算法列表，包含 CLR
M_candidates = [2, 3, 4, 5, 6]
# good seed: 380776, 458676

def run_single_experiment(sample_frac, pilot_frac):

    # --------------------------------------------------
    # 0. Load data
    # --------------------------------------------------
    seed = np.random.randint(0, 1_000_000)
    X, y, D = load_criteo(sample_frac=sample_frac, seed=seed)

    # --------------------------------------------------
    # 1–3. pilot + outcome models
    # --------------------------------------------------
    (
        X_pilot, X_impl,
        D_pilot, D_impl,
        y_pilot, y_impl,
        mu1_pilot_model, mu0_pilot_model,
        e_pilot, Gamma_pilot
    ) = prepare_pilot_impl(X, y, D, pilot_frac=pilot_frac)

    # storage for output
    results = {
        "seed": int(seed),
        "e_pilot": float(e_pilot)
    }

    # --------------------------------------------------
    # Baselines — Always run
    # --------------------------------------------------
    seg_labels_all_treat = np.zeros(len(X_impl), dtype=int)
    value_all_treat = evaluate_policy(
        X_impl, D_impl, y_impl,
        seg_labels_all_treat,
        mu1_pilot_model, mu0_pilot_model,
        np.array([1])
    )
    results["all_treat"] = float(value_all_treat["value_mean"])

    seg_labels_all_control = np.zeros(len(X_impl), dtype=int)
    value_all_control = evaluate_policy(
        X_impl, D_impl, y_impl,
        seg_labels_all_control,
        mu1_pilot_model, mu0_pilot_model,
        np.array([0])
    )
    results["all_control"] = float(value_all_control["value_mean"])

    seg_labels_random = np.random.binomial(1, e_pilot, size=len(X_impl))
    value_random = evaluate_policy(
        X_impl, D_impl, y_impl,
        seg_labels_random,
        mu1_pilot_model, mu0_pilot_model,
        np.array([0, 1])
    )
    results["random"] = float(value_random["value_mean"])


    # --------------------------------------------------
    # 4a. KMeans
    # --------------------------------------------------
    if "kmeans" in ALGO_LIST:
        t0 = time.perf_counter()
        kmeans_seg, seg_labels_pilot_kmeans, best_K_kmeans = run_kmeans_segmentation(
            X_pilot, D_pilot, y_pilot, K_candidates=M_candidates
        )
        action_kmeans = estimate_segment_policy(
            X_pilot, y_pilot, D_pilot, seg_labels_pilot_kmeans
        )
        seg_labels_impl_kmeans = kmeans_seg.assign(X_impl)
        value_kmeans = evaluate_policy(
            X_impl, D_impl, y_impl,
            seg_labels_impl_kmeans,
            mu1_pilot_model, mu0_pilot_model,
            action_kmeans
        )
        t1 = time.perf_counter()
        results["kmeans"] = float(value_kmeans["value_mean"])
        results["time_kmeans"] = float(t1 - t0)   # 👈 记录时间（单位秒）
        print(
        f"KMeans - Segments: {len(np.unique(seg_labels_pilot_kmeans))}, "
        f"Actions: {action_kmeans}",
        )


    # --------------------------------------------------
    # 4b. GMM
    # --------------------------------------------------
    if "gmm" in ALGO_LIST:
        t0 = time.perf_counter()
        gmm_seg, seg_labels_pilot_gmm, best_K_gmm = run_gmm_segmentation(
            X_pilot, D_pilot, y_pilot, K_candidates=M_candidates,
        )
        action_gmm = estimate_segment_policy(
            X_pilot, y_pilot, D_pilot, seg_labels_pilot_gmm
        )
        seg_labels_impl_gmm = gmm_seg.assign(X_impl)
        value_gmm = evaluate_policy(
            X_impl, D_impl, y_impl,
            seg_labels_impl_gmm,
            mu1_pilot_model, mu0_pilot_model,
            action_gmm
        )
        t1 = time.perf_counter()
        results["gmm"] = float(value_gmm["value_mean"])
        results["time_gmm"] = float(t1 - t0)   # 👈 记录时间（单位秒）
        print(
        f"GMM - Segments: {len(np.unique(seg_labels_pilot_gmm))}, "
        f"Actions: {action_gmm}",
        )


    # --------------------------------------------------
    # 4c. CLR
    # --------------------------------------------------
    if "clr" in ALGO_LIST:
        t0 = time.perf_counter()
        clr_seg, seg_labels_pilot_clr, best_K_clr = run_clr_segmentation(
            X_pilot, D_pilot, y_pilot,
            K_candidates=M_candidates,
            kmeans_coef=0.1,
            num_tries=8,
        )
        action_clr = estimate_segment_policy(
            X_pilot, y_pilot, D_pilot, seg_labels_pilot_clr
        )
        seg_labels_impl_clr = clr_seg.assign(X_impl)
        value_clr = evaluate_policy(
            X_impl, D_impl, y_impl,
            seg_labels_impl_clr,
            mu1_pilot_model, mu0_pilot_model,
            action_clr
        )
        t1 = time.perf_counter()
        results["clr"] = float(value_clr["value_mean"])
        results["time_clr"] = float(t1 - t0)   # 👈 记录时间（单位秒）
        print(
        f"CLR - Segments: {len(np.unique(seg_labels_pilot_clr))}, "
        f"Actions: {action_clr}",
        )


    # --------------------------------------------------
    # 5–6. DAST
    # --------------------------------------------------
    if "dast" in ALGO_LIST:
        t0 = time.perf_counter()
        (
            tree_final,
            seg_labels_pilot_dast,
            best_M_dast,
            best_action_dast_train,
        ) = run_dast_dams(
            X_pilot, D_pilot, y_pilot,
            mu1_pilot_model, mu0_pilot_model,
            e_pilot, Gamma_pilot,
            train_frac=train_frac,
            M_candidates=M_candidates,
            min_leaf_size=5,
        )
        action_dast = estimate_segment_policy(
            X_pilot, y_pilot, D_pilot, seg_labels_pilot_dast
        )
        seg_labels_impl_dast = tree_final.assign(X_impl)
        value_dast = evaluate_policy(
            X_impl, D_impl, y_impl,
            seg_labels_impl_dast,
            mu1_pilot_model, mu0_pilot_model,
            action_dast
        )
        t1 = time.perf_counter()
        results["dast"] = float(value_dast["value_mean"])
        results["time_dast"] = float(t1 - t0)   # 👈 记录时间（单位秒）
        print(
        f"DAST - Segments: {len(np.unique(seg_labels_pilot_dast))}, "
        f"Actions: {action_dast}",
        )
    
    # MST
    if "mst" in ALGO_LIST:
        t0 = time.perf_counter()
        tree_mst, seg_labels_pilot_mst, best_M_mst, action_mst = run_mst_dams(
            X_pilot, D_pilot, y_pilot,
            mu1_pilot_model, mu0_pilot_model,
            e_pilot,
            train_frac=0.5,
            M_candidates=M_candidates,
            min_leaf_size=5,
        )

        print(
        f"MST - Segments: {len(np.unique(seg_labels_pilot_mst))}, "
        f"Actions: {action_mst}",
        )

        seg_labels_impl_mst = tree_mst.assign(X_impl)

        value_mst = evaluate_policy(
            X_impl, D_impl, y_impl,
            seg_labels_impl_mst,
            mu1_pilot_model, mu0_pilot_model,
            action_mst,
        )
        t1 = time.perf_counter()
        results["mst"] = float(value_mst["value_mean"])
        results["time_mst"] = float(t1 - t0)  
        

    # Policytree (R based)
    if "policytree" in ALGO_LIST:
        t0 = time.perf_counter()
        policy_seg, seg_labels_pilot_policy, best_M, best_M_policy = run_policytree_segmentation(
            X_pilot, D_pilot, y_pilot,
            mu1_pilot_model, mu0_pilot_model, e_pilot,
            train_frac=0.7,
            M_candidates=M_candidates,
        )
        action_policy = estimate_segment_policy(
            X_pilot, y_pilot, D_pilot, seg_labels_pilot_policy
        )
        seg_labels_impl_policy = policy_seg.assign(X_impl)
        value_policy = evaluate_policy(
            X_impl, D_impl, y_impl,
            seg_labels_impl_policy,
            mu1_pilot_model, mu0_pilot_model,
            action_policy, e_pilot
        )
        t1 = time.perf_counter()
        results["policytree"] = float(value_policy["value_mean"])
        results["time_policytree"] = float(t1 - t0)
        print(
            f"PolicyTree - Segments: {len(np.unique(seg_labels_pilot_policy))}, "
            f"Actions: {action_policy}, Time: {t1 - t0:.2f} seconds"
        )



    # --------------------------------------------------
    # 输出 summary
    # --------------------------------------------------
    print("\nResult for this run:")
    for k, v in results.items():
        print(f"{k:15s}: {v}")

    return results

def run_multiple_experiments(
    N_sim,
    sample_frac,
    pilot_frac,
    out_path,
):
    experiment_data = {
        "params": {
            "sample_frac": sample_frac,
            "pilot_frac": pilot_frac,
            "train_frac": train_frac,
            "N_sim": N_sim
        },
        "results": []  # 用来存每次 run 的结果
    }

    print("\n" + "=" * 60)
    print(f"STARTING SIMULATIONS: N_sim = {N_sim}")
    print("=" * 60)

    for s in range(N_sim):
        try:
            res = run_single_experiment(
                sample_frac=sample_frac,
                pilot_frac=pilot_frac,
            )
        
            experiment_data["results"].append(res)

            # 每轮覆盖保存
            with open(out_path, "wb") as f:
                pickle.dump(experiment_data, f)
            
            print(f'[SIM {len(experiment_data["results"])}/{N_sim}] saved → {out_path}')
            print("-" * 60)
            
        
        except:
            # print reason for error
            import traceback
            traceback.print_exc()  # 打印完整错误堆栈，便于定位
            continue

        

    print("\nALL SIMULATIONS DONE.")
    print(f"Results saved in '{out_path}'")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run multiple Criteo segmentation experiments"
    )

    parser.add_argument(
        "--outpath",
        type=str,
        default=None,
        help="Output pkl path"
    )

    args = parser.parse_args()

    pilot_frac = 0.5  # 50% data for pilot
    train_frac = 0.7  # 70% pilot for training
    
    run_multiple_experiments(
        N_sim=1,
        sample_frac=0.002,
        pilot_frac=pilot_frac,
        out_path=args.outpath,
    )
