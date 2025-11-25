import numpy as np
from outcome_model import predict_mu


import numpy as np
from outcome_model import predict_mu

import numpy as np
from outcome_model import predict_mu


def evaluate_policy(
        X_impl, D_impl, y_impl,
        seg_labels_impl,
        mu1_model, mu0_model,
        action,
        e_pilot=None,   # 保留参数占位，但这里不会用到
    ):
    """
    Policy evaluation（非 DR 版本）

    公式：
        v_i = y_i                 if D_i == a_i
            = μ_{a_i}(x_i)        if D_i != a_i

    这里 μ_{a_i}(x_i) 由 outcome model 给出：
        a_i = 1 -> μ_1(x_i)
        a_i = 0 -> μ_0(x_i)
    """
    # 1. 预测 μ1(x), μ0(x)
    mu1 = predict_mu(mu1_model, X_impl)
    mu0 = predict_mu(mu0_model, X_impl)

    # 2. 根据 segment label 得到 policy 对每个样本的 action: a_i
    a = action[seg_labels_impl].astype(int)

    # 3. 对应 policy action 的预测 μ_{a_i}(x_i)
    mu_a = np.where(a == 1, mu1, mu0)

    # 4. 套用公式：
    #    D_i = a_i → 用真实 y_i
    #    D_i != a_i → 用 μ_{a_i}(x_i)
    v = np.where(D_impl == a, y_impl, mu_a)

    return {
        "value_mean": float(v.mean()),
        "value_sum": float(v.sum()),
    }



def evaluate_policy_dr(
        X_impl, D_impl, y_impl,
        seg_labels_impl,
        mu1_model, mu0_model,
        action,
        e_pilot
    ):
    """
    Doubly-robust off-policy evaluation for a binary treatment policy.

    X_impl:        实施阶段的特征 (n, d)
    D_impl:        实施阶段实际 treatment (0/1)  (n,)
    y_impl:        实际 outcome                 (n,)
    seg_labels_impl: 每个样本对应的 segment id   (n,)
    mu1_model, mu0_model:  outcome model
    action:        对每个 segment 的 policy 决策 (0/1)，用 seg_labels_impl 来 index
    e_pilot:       实施阶段 treat 的概率（如果是完全随机，常数即可；
                   如果你有更细的 e_i(X)，可以传向量进来，代码也兼容）
    """
    # 1. 预测 μ1, μ0
    mu1 = predict_mu(mu1_model, X_impl)
    mu0 = predict_mu(mu0_model, X_impl)

    # 2. 处理 propensity，可以是标量也可以是向量
    if np.isscalar(e_pilot):
        e_impl = np.full_like(y_impl, fill_value=e_pilot, dtype=float)
    else:
        e_impl = np.asarray(e_pilot, dtype=float)
        assert e_impl.shape == y_impl.shape

    e_impl = np.clip(e_impl, 1e-6, 1 - 1e-6)

    # 3. policy 的 action a_i = π(X_i)
    a = action[seg_labels_impl].astype(int)

    # 4. 构造 DR 估计
    mu_d  = D_impl * mu1 + (1 - D_impl) * mu0      # μ_{D_i}(X_i)
    mu_pi = a * mu1 + (1 - a) * mu0                # μ_{π(X_i)}(X_i)

    p_a = a * e_impl + (1 - a) * (1 - e_impl)      # p_{π(X_i)}(X_i)
    indicator = (D_impl == a).astype(float)

    v = mu_pi + indicator / p_a * (y_impl - mu_d)

    return {
        "value_mean": float(v.mean()),
        "value_sum": float(v.sum()),
    }
    
    
import numpy as np


def evaluate_policy_ips(
        X_impl, D_impl, y_impl,
        seg_labels_impl,
        mu1_model, mu0_model,
        action,
        e_pilot
    ):
    """
    纯 IPW policy evaluation，只用 y，不用 outcome model / Gamma。

    公式：
        V_hat = (1/n) * sum_i [ 1{D_i = a_i} / p_{a_i} * y_i ]

    其中 p_{a_i} = e_pilot (a_i=1) or 1-e_pilot (a_i=0)
    """

    # 1. policy 决策 a_i
    a = action[seg_labels_impl].astype(int)

    # 2. 处理 propensity（可以是标量或向量）
    if np.isscalar(e_pilot):
        e = np.full_like(y_impl, e_pilot, dtype=float)
    else:
        e = np.asarray(e_pilot, dtype=float)
        assert e.shape == y_impl.shape

    e = np.clip(e, 1e-6, 1 - 1e-6)

    # 3. 对应 policy action 的概率 p_{a_i}
    p_a = a * e + (1 - a) * (1 - e)   # e if a=1 else 1-e

    # 4. IPW 权重，只在 D==a 时非 0
    indicator = (D_impl == a).astype(float)
    w = indicator / p_a

    # 5. 估计值
    contrib = w * y_impl   # 每个样本对 V 的贡献
    value_mean = float(contrib.mean())

    return {
        "value_mean": value_mean,
        "value_sum": float(value_mean * len(y_impl)),  # 按全体样本数量 scale
    }



def evaluate_policy_for_random_baseline(X_impl, D_impl, y_impl,
                           a_random,
                           mu1_model, mu0_model,
                           e_pilot):
    """
    Evaluate an individual-level random policy using DR (dual) estimator.

    Parameters
    ----------
    X_impl, D_impl, y_impl : implementation data
    a_random : np.ndarray, shape (N_impl,)
        Individual-level action recommended by the random policy (0/1).
    mu1_model, mu0_model : outcome models fitted on pilot
    e_pilot : float
        Propensity (treatment probability) used in pilot / DR construction
    """
    # Predict potential outcomes under 1 and 0
    mu1_impl = predict_mu(mu1_model, X_impl)
    mu0_impl = predict_mu(mu0_model, X_impl)

    # DR scores for factual treatment assignment
    Gamma1 = mu1_impl + (D_impl / e_pilot) * (y_impl - mu1_impl)
    Gamma0 = mu0_impl + ((1 - D_impl) / (1 - e_pilot)) * (y_impl - mu0_impl)

    # Dual DR estimator: use y if action matches, otherwise use counterfactual Gamma
    match = (D_impl == a_random)
    v = np.empty_like(y_impl, dtype=float)

    v[match] = y_impl[match]
    v[~match] = np.where(
        a_random[~match] == 1,
        Gamma1[~match],
        Gamma0[~match]
    )

    return {
        "value_mean": v.mean(),
        "value_sum": v.sum(),
    }





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
    print("\n" + "="*60)
    print("STEP 9: Evaluating policies on implementation data")
    print("="*60)

    # Baseline 1: All Treat
    seg_labels_all_treat = np.zeros(len(X_impl), dtype=int)  # All in segment 0
    action_all_treat = np.array([1])  # Action = 1 for all
    value_all_treat = evaluate_policy(
        X_impl, D_impl, y_impl,
        seg_labels_all_treat,
        mu1_pilot_model, mu0_pilot_model,
        action_all_treat,
        e_pilot
    )

    # Baseline 2: All Control
    seg_labels_all_control = np.zeros(len(X_impl), dtype=int)  # All in segment 0
    action_all_control = np.array([0])  # Action = 0 for all
    value_all_control = evaluate_policy(
        X_impl, D_impl, y_impl,
        seg_labels_all_control,
        mu1_pilot_model, mu0_pilot_model,
        action_all_control,
        e_pilot
    )

    # Baseline 3: Random (Bernoulli(e_pilot))
    a_random_impl = np.random.binomial(1, e_pilot, size=len(X_impl))
    value_random = evaluate_policy_for_random_baseline(
        X_impl, D_impl, y_impl,
        a_random_impl,
        mu1_pilot_model, mu0_pilot_model,
        e_pilot
    )

    # KMeans policy
    value_kmeans = evaluate_policy(
        X_impl, D_impl, y_impl,
        seg_labels_impl_kmeans,
        mu1_pilot_model, mu0_pilot_model,
        action_kmeans,
        e_pilot
    )
    # GMM
    print("Evaluating GMM policy...")
    val_gmm = evaluate_policy(
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
        X_impl, D_impl, y_impl,
        seg_labels_impl_dast,
        mu1_pilot_model, mu0_pilot_model,
        action_dast,
        e_pilot
    )

    # ================== 主结果 ==================
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Baseline - All Control: {value_all_control['value_mean']:.6f}")
    print(f"Baseline - Random:      {value_random['value_mean']:.6f}")
    print(f"Baseline - All Treat:   {value_all_treat['value_mean']:.6f}")
    print(f"KMeans policy:          {value_kmeans['value_mean']:.6f}")
    print(f"DAST   policy:          {value_dast['value_mean']:.6f}")
    print("="*60)

    # ================== 相对提升（可选，但强烈推荐） ==================
    print("\n" + "="*60)
    print("RELATIVE IMPROVEMENT of DAST")
    print("="*60)

    baselines = {
        "All Control": value_all_control,
        "Random":      value_random,
        "All Treat":   value_all_treat,
        "KMeans":      value_kmeans,
        "GMM":         val_gmm,
    }

    v_dast = value_dast["value_mean"]

    def safe_div(a, b):
        return a / (b + 1e-12)

    for name, val in baselines.items():
        v_base = val["value_mean"]
        abs_gain = v_dast - v_base
        rel_gain = safe_div(abs_gain, abs(v_base)) * 100.0

        print(f"{name:12s}:  "
            f"abs Δ = {abs_gain:+.6f},   "
            f"relative = {rel_gain:+.2f}%")

    print("="*60)


