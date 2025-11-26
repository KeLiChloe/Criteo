# test_dast.py
import numpy as np
import matplotlib.pyplot as plt

from dast import DASTree  # 确保 dast.py 在同一目录下


# -----------------------------------------------------
# 1. 模拟数据
# -----------------------------------------------------
def simulate_data(
    n_per_cluster=300,
    d=1,
    sigma_x=2,
    sigma_eps=0.5,
    seed=42,
):
    """
    构造一个简单的 1D toy 示例：
    - 3 个簇，每个有自己的 x mean, alpha, beta, tau
    - D ~ Bern(0.5)
    - y = alpha_k + beta_k * x + tau_k * D + noise
    - Gamma 使用真潜在结果 (oracle)
    """
    rng = np.random.RandomState(seed)

    K = 3  # 3 clusters
    means_x = [-5.0, 5, 10.0]
    alphas = rng.normal(0.0, 1.0, size=K)
    betas = rng.normal(1.0, 0.3, size=K)
    taus = [2.0, -1.0, -3.0]  # treatment effects

    print("🌱 True cluster parameters:")
    for k in range(K):
        print(f"  Cluster {k}: mean_x={means_x[k]:.2f}, "
              f"alpha={alphas[k]:.2f}, beta={betas[k]:.2f}, tau={taus[k]:.2f}")

    X_list, y_list, D_list, Z_list, mu0_list, mu1_list = [], [], [], [], [], []

    for k in range(K):
        x_k = rng.normal(means_x[k], sigma_x, size=(n_per_cluster, d))
        D_k = rng.binomial(1, 0.5, size=n_per_cluster)

        x_scalar = x_k[:, 0]
        mu0_k = alphas[k] + betas[k] * x_scalar
        mu1_k = mu0_k + taus[k]
        eps_k = rng.normal(0, sigma_eps, size=n_per_cluster)

        y_k = mu0_k + taus[k] * D_k + eps_k

        X_list.append(x_k)
        y_list.append(y_k)
        D_list.append(D_k)
        Z_list.append(np.full(n_per_cluster, k))
        mu0_list.append(mu0_k)
        mu1_list.append(mu1_k)

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    D = np.concatenate(D_list)
    Z = np.concatenate(Z_list)
    mu0 = np.concatenate(mu0_list)
    mu1 = np.concatenate(mu1_list)

    Gamma = np.vstack([mu0, mu1]).T

    return X, y, D, Z, Gamma



# -----------------------------------------------------
# 3. 收集 split
# -----------------------------------------------------
def collect_split_thresholds(tree):
    thresholds = []
    def dfs(node):
        if node is None:
            return
        if not node.is_leaf:
            thresholds.append(node.split_threshold)
            dfs(node.left)
            dfs(node.right)
    dfs(tree.root)
    return thresholds


# -----------------------------------------------------
# 4. 画图帮助函数（marker 区分 treatment）
# -----------------------------------------------------
def scatter_with_treatment(X, y, labels, D, title):
    """
    X: (N,1)
    y: (N,)
    labels: cluster 或 segment id
    D: treatment (0/1)
    """
    plt.figure(figsize=(8,4))

    # 控制组：圆点
    plt.scatter(
        X[D==0, 0], y[D==0],
        c=labels[D==0],
        cmap="tab10",
        alpha=0.7,
        s=25,
        marker="o",
        label="D=0 (control)"
    )

    # 处理组：叉号
    plt.scatter(
        X[D==1, 0], y[D==1],
        c=labels[D==1],
        cmap="tab10",
        alpha=0.9,
        s=35,
        marker="x",
        label="D=1 (treated)"
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)


# -----------------------------------------------------
# 5. 主程序
# -----------------------------------------------------
def main():
    # 1. 数据模拟
    X, y, D, Z, Gamma = simulate_data()
    print("\n📌 Simulated data generated.")

    # 2. 可视化原始数据（真实簇）
    scatter_with_treatment(
        X, y, labels=Z, D=D,
        title="Simulated Data: True Clusters (marker = treatment)"
    )

    # 3. 构造 DAST
    d_full = X.shape[1]
    # Generate candidate thresholds (midpoints between unique values)
    # Generate candidate thresholds (midpoints between unique values)
    bins = 64  # 每个特征最多 64 个候选阈值
    H_full = {}

    for j in range(d_full):
        col = X[:, j]
        # 去掉nan的话可以先 col = col[~np.isnan(col)]
        unique_values = np.unique(col)

        if len(unique_values) <= 1:
            H_full[j] = unique_values
        else:
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

    tree = DASTree(
        x=X,
        y=y,
        D=D,
        gamma=Gamma,
        candidate_thresholds=H_full,
        min_leaf_size=10,
        max_depth=2,
        epsilon=0.0,
    )

    print("\n🌳 Building DAST tree ...")
    tree.build()
    tree.prune_to_M(2)

    seg_labels = tree.assign(X)
    splits = collect_split_thresholds(tree)
    print(f"DAST split thresholds = {splits}")

    # 4. 可视化 DAST 结果
    scatter_with_treatment(
        X, y, labels=seg_labels, D=D,
        title="DAST Segmentation (M=2) with Treatment Marker"
    )

    # 画分割线
    for thr in splits:
        plt.axvline(thr, color="red", linestyle="--", lw=2)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
