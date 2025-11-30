# scoring.py
import numpy as np
from outcome_model import predict_mu


from sklearn.metrics import silhouette_score

def kmeans_silhouette_score(seg_model, X_pilot):
    """
    KMeans segmentation scoring using Silhouette Score.

    Parameters
    ----------
    seg_model : KMeansSeg
        已经调用 seg_model.fit(X_pilot) 的 KMeans segmentation 模型
    X_pilot : np.ndarray
        Feature matrix for pilot dataset  
    D_pilot, y_pilot : not used
        只是为了接口统一（和 DAST 的 score 保持一致）

    Returns
    -------
    float
        Silhouette Score (越大越好)
    """
    labels = seg_model.assign(X_pilot)

    return silhouette_score(X_pilot, labels)

    


def dams_score(seg_model, X_val, D_val, y_val, Gamma_val, action):
    """
    Decision-Aware Model Selection 的 scoring 函数（Algorithm 3）.

    参数
    ----
    seg_model : 已经在 train_seg 上 fit 好的 segmentation 模型
                需要有 .assign(X) 方法
    X_val, D_val, y_val : validation 集
    mu1_model, mu0_model : 在 pilot 上 fit 的 outcome models
    e : pilot 阶段的 treatment 比例（propensity）
    action : ndarray, shape (M,)
             每个 segment m 的动作 a_m，必须是在 train 阶段学好的

    返回
    ----
    V_val : float
        DR 估计的 policy value 在 validation 集上的平均值
    """
    # 1) segmentation: assign each i to segment m
    labels = seg_model.assign(X_val)        # shape (N_val,)


    # 3) 应用“之前学好的” segment-level action
    a_i = action[labels]                    # π^{C_M}(i)

    # 4) DR policy value:
    #    v̂_i = y_i        if D_i == a_i
    #         = Γ_{i,a_i}  otherwise
    v_hat = np.empty_like(y_val, dtype=float)

    mask_match = (D_val == a_i)
    mask_mismatch = ~mask_match

    v_hat[mask_match] = y_val[mask_match]
    # 对 mismatch 的：根据 a_i 选 Gamma1 或 Gamma0
    v_hat[mask_mismatch] = np.where(
        a_i[mask_mismatch] == 1,
        Gamma_val[mask_mismatch, 1],
        Gamma_val[mask_mismatch, 0],
    )

    return float(v_hat.mean())
