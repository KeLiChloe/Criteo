# mst.py

import copy
import numpy as np
from sklearn.linear_model import LinearRegression


def _build_design_matrix(X: np.ndarray, D: np.ndarray) -> np.ndarray:
    """
    构造线性回归的设计矩阵:
        y ~ 1 + X + D
    X: (n, d), D: (n,) 或 (n,1)
    返回: (n, 1 + d + 1)
    """
    if D.ndim == 2 and D.shape[1] == 1:
        D_vec = D.ravel()
    else:
        D_vec = D
    n, d = X.shape
    intercept = np.ones((n, 1), dtype=float)
    D_col = D_vec.reshape(-1, 1)
    return np.hstack([intercept, X, D_col])


def compute_residual_value(X: np.ndarray,
                           y: np.ndarray,
                           D: np.ndarray,
                           indices: np.ndarray) -> float:
    """
    在一个 node 里，用 OLS 拟合:
        y ~ 1 + X + D
    返回 SSE（总残差和），SSE 越小越好。
    为了让接口和 DAST 一致，我们在 MSTree 里使用 value = -SSE。
    """
    if len(indices) == 0:
        return 0.0

    X_m = X[indices]
    y_m = y[indices]
    D_m = D[indices]

    X_design = _build_design_matrix(X_m, D_m)
    model = LinearRegression(fit_intercept=False)
    model.fit(X_design, y_m)

    y_pred = model.predict(X_design)
    residuals = y_m - y_pred
    sse = float(np.sum(residuals ** 2))

    # 加一点极小噪声，避免完全 tie
    sse += np.random.rand() * 1e-9
    return sse


class MSTNode:
    def __init__(self, indices, depth=0):
        self.indices = indices       # np.ndarray of row indices
        self.depth = depth
        self.value = None            # 我们存的是 value = -SSE

        # split info
        self.split_feature = None
        self.split_threshold = None
        self.left = None
        self.right = None
        self.is_leaf = True

        # after prune
        self.segment_id = None

    def prune(self):
        self.left = None
        self.right = None
        self.is_leaf = True


class MSTree:
    """
    MST segmentation tree.

    和 DASTree 一样的接口：
        - build()
        - prune_to_M(M)
        - assign(X)
    唯一差别是：节点价值用的是 -SSE（线性回归残差）。
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        D: np.ndarray,
        candidate_thresholds,
        min_leaf_size: int,
        max_depth: int,
        epsilon: float = 0.0,
    ):
        """
        x : (N, d)
        y : (N,)
        D : (N,)
        candidate_thresholds: dict[int -> 1D array] 或 list，和 DAST 一致
        """
        self.x = x
        self.y = y
        self.D = D
        self.H = candidate_thresholds
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth
        self.epsilon = epsilon

        self.root: MSTNode | None = None
        self.leaf_nodes: list[MSTNode] = []

        # for pruning float compare
        self.tolerance_pruning = 1e-8

    # ======================================================
    # Public API
    # ======================================================
    def build(self):
        N = self.x.shape[0]
        all_indices = np.arange(N)
        self.root = self._grow_node(all_indices, depth=0)
    
    def copy(self):
        return copy.deepcopy(self)

    def prune_to_M(self, M: int):
        """
        和 DAST 的 prune_to_M 完全同构，只是 node.value 换成了 -SSE。
        也就是：我们优先保留“减小 SSE 最多”的 split。
        """
        if self.root is None:
            raise RuntimeError("Call build() before prune_to_M().")

        current_leaves = self._get_leaf_nodes()

        # 如果一开始叶子少于 M，就直接返回（和你现在 DAST 版本一致的宽松做法）
        if len(current_leaves) <= M:
           print(f"Warning: current leaf count {len(current_leaves)} <= target M={M}. No pruning performed.")

        while len(current_leaves) > M:
            prunable_nodes = self._get_internal_nodes_with_leaf_children()
            if not prunable_nodes:
                break

            best_node = None
            # 注意：这里 gain 越大说明 split 越差（因为 value=-SSE）
            # gain = parent_value - (left + right)
            best_gain = -np.inf
            best_variance_increase = np.inf   # tie-breaking 用 covariate variance

            for node in prunable_nodes:
                gain = self._compute_pruning_gain(node)
                if gain > best_gain + self.tolerance_pruning:
                    best_gain = gain
                    best_node = node
                    best_variance_increase = self._compute_variance_after_pruning(node)
                elif abs(gain - best_gain) <= self.tolerance_pruning:
                    # tie-breaking: variance increase 较小者优先
                    var_inc = self._compute_variance_after_pruning(node)
                    if var_inc < best_variance_increase:
                        best_variance_increase = var_inc
                        best_node = node

            if best_node is None:
                break

            best_node.prune()
            current_leaves = self._get_leaf_nodes()

        self.leaf_nodes = current_leaves
        for seg_id, node in enumerate(self.leaf_nodes):
            node.segment_id = seg_id

    def assign(self, X: np.ndarray) -> np.ndarray:
        """
        和 DASTree.assign 一样：对任意 X 返回 segment_id。
        """
        if self.root is None:
            raise RuntimeError("Call build() before assign().")

        labels = np.empty(X.shape[0], dtype=int)
        for i, x_i in enumerate(X):
            node = self.root
            while not node.is_leaf:
                j = node.split_feature
                t = node.split_threshold
                if x_i[j] <= t:
                    node = node.left
                else:
                    node = node.right
            labels[i] = node.segment_id
        return labels

    # ======================================================
    # Growing
    # ======================================================
    def _grow_node(self, indices: np.ndarray, depth: int) -> MSTNode:
        node = MSTNode(indices=indices, depth=depth)

        # 节点 value 定义为 -SSE，这样“越大越好”，统一成和 DAST 一样的方向
        sse = compute_residual_value(self.x, self.y, self.D, indices)
        node.value = -sse

        if depth == self.max_depth:
            self.leaf_nodes.append(node)
            return node

        best_gain = -np.inf
        best_split = None

        d = self.x.shape[1]

        for j in range(d):
            thresholds = self.H[j] if isinstance(self.H, dict) else self.H[j]
            for t in thresholds:
                left_idx = indices[self.x[indices, j] <= t]
                right_idx = indices[self.x[indices, j] > t]

                if not (self._check_leaf_constraints(left_idx) and
                        self._check_leaf_constraints(right_idx)):
                    continue

                left_sse = compute_residual_value(self.x, self.y, self.D, left_idx)
                right_sse = compute_residual_value(self.x, self.y, self.D, right_idx)

                left_val = -left_sse
                right_val = -right_sse

                gain = left_val + right_val - node.value  # “价值增量”

                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_split = (j, t, left_idx, right_idx)

        # gain 太小可以用 epsilon 阈值截断（可选）
        if best_split is None or best_gain <= self.epsilon:
            self.leaf_nodes.append(node)
            return node

        # 执行 split
        node.is_leaf = False
        feature, threshold, left_idx, right_idx = best_split
        node.split_feature = feature
        node.split_threshold = threshold
        node.left = self._grow_node(left_idx, depth + 1)
        node.right = self._grow_node(right_idx, depth + 1)

        return node

    # ======================================================
    # Helpers: variance / constraints / pruning
    # ======================================================
    def _check_leaf_constraints(self, indices) -> bool:
        if len(indices) == 0:
            return False
        D_sub = self.D[indices]
        n1 = np.sum(D_sub == 1)
        n0 = np.sum(D_sub == 0)
        return (n1 >= self.min_leaf_size) and (n0 >= self.min_leaf_size)

    def _compute_covariate_variance(self, indices) -> float:
        if len(indices) < 2:
            return 0.0
        return np.var(self.x[indices], axis=0, ddof=1).sum()

    def _compute_variance_after_pruning(self, node) -> float:
        """
        用在 prune 阶段的 tie-breaking：
        计算把两个叶子 merge 成 parent 后，X 的方差增加量。
        """
        if node.left is None or node.right is None:
            return np.inf

        merged_indices = node.indices
        left_indices = node.left.indices
        right_indices = node.right.indices

        merged_var = self._compute_covariate_variance(merged_indices)
        left_var = self._compute_covariate_variance(left_indices)
        right_var = self._compute_covariate_variance(right_indices)

        n_total = len(merged_indices)
        if n_total == 0:
            return np.inf

        n_left = len(left_indices)
        n_right = len(right_indices)
        weighted_before = (n_left * left_var + n_right * right_var) / n_total

        return merged_var - weighted_before

    def _compute_pruning_gain(self, node) -> float:
        """
        pruning gain = parent_value - (left_value + right_value)
        这里 value = -SSE:
          - 如果 split 很好（S_parent >> S_L+S_R），则 gain 很负（不想 prune）
          - 如果 split 很差（S_parent <= S_L+S_R），则 gain >=0（优先 prune）
        prune_to_M 里会选 gain 最大的节点去剪枝。
        """
        if node.left is None or node.right is None:
            return -np.inf

        parent_val = node.value
        left_val = node.left.value
        right_val = node.right.value
        return parent_val - (left_val + right_val)

    # ======================================================
    # Tree traversal utilities
    # ======================================================
    def _get_leaf_nodes(self):
        return self._gather_nodes(self.root, lambda n: n.is_leaf)

    def _get_internal_nodes_with_leaf_children(self):
        def cond(n: MSTNode):
            return (
                (not n.is_leaf) and
                (n.left is not None) and
                (n.right is not None) and
                n.left.is_leaf and
                n.right.is_leaf
            )
        return self._gather_nodes(self.root, cond)

    def _gather_nodes(self, node, condition):
        if node is None:
            return []
        res = [node] if condition(node) else []
        res += self._gather_nodes(node.left, condition)
        res += self._gather_nodes(node.right, condition)
        return res
