import numpy as np
from outcome_model import predict_mu


import numpy as np
from outcome_model import predict_mu


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
