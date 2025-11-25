# exp.py  —— Criteo 上对比 KMeans / GMM / DAST 的完整实验脚本

import numpy as np


from data_utils import load_criteo, prepare_pilot_impl
from evaluation import evaluate_all_policies
from estimation import estimate_segment_policy    # 版本: estimate_segment_policy(X, y, D, seg_labels)
from segmentation import run_dast_dams, run_kmeans_segmentation, run_gmm_segmentation

def main():
    # 0. load data
    seed = np.random.randint(0, 1_000_000)
    print(f"Using random seed: {seed}")
    X, y, D = load_criteo(sample_frac=0.03, 
                          random_state=seed)
    # (0.01) 497262, 330552, 758598, 705628
    
    # show first three rows
    print("\nFirst three rows of the dataset:")
    for i in range(3):
        print(f"Row {i+1}: X={X[i]}, D={D[i]}, y={y[i]}")
        

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
        Gamma_pilot,
    ) = prepare_pilot_impl(X, y, D, pilot_frac=0.4)

    # 4a. KMeans
    kmeans_seg, seg_labels_pilot_kmeans, best_K_kmeans = run_kmeans_segmentation(
        X_pilot, D_pilot, y_pilot, K_candidates=[2, 3, 4]
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
        train_frac=0.5,
        M_candidates=(2, 3, 4, 5, 6),
        min_leaf_size=5,
    )

    # 7. 估计 segment-level policy（diff-in-means on y）
    print("\n" + "=" * 60)
    print("STEP 7: Estimating segment policies")
    print("=" * 60)

    tau_kmeans, action_kmeans = estimate_segment_policy(
        X_pilot, y_pilot, D_pilot, seg_labels_pilot_kmeans
    )
    print(
        f"KMeans - Segments: {len(np.unique(seg_labels_pilot_kmeans))}, "
        f"Actions: {action_kmeans}"
    )

    tau_gmm, action_gmm = estimate_segment_policy(
        X_pilot, y_pilot, D_pilot, seg_labels_pilot_gmm
    )
    print(
        f"GMM    - Segments: {len(np.unique(seg_labels_pilot_gmm))}, "
        f"Actions: {action_gmm}"
    )

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
