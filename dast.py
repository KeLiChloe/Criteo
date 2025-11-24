"""
DAST (Decision-Aware Segmentation Tree)

Implements Algorithm 1 (BuildDecisionAwareTree) and Algorithm 2 (PostPruneTree),
but只做“分段”：建树 + 剪枝到 M 个叶子 + 返回 segment_id。

- 输入：X, y, D, Gamma
- Gamma: doubly robust score matrix, shape (N, 2), column 0 对应 a=0, column 1 对应 a=1
- 候选阈值: H = {H_j}_j, H_j 是 feature j 上的候选 split points
"""

import numpy as np


class DASTNode:
    """Node in a DAST tree."""

    def __init__(self, indices, depth=0):
        # indices: np.ndarray of row indices in training data
        self.indices = indices
        self.depth = depth
        self.value = None  # V_hat(L)

        # split info
        self.split_feature = None
        self.split_threshold = None
        self.left = None
        self.right = None
        self.is_leaf = True

        # segment id after pruning
        self.segment_id = None

    def prune(self):
        """Collapse children and make this node a leaf."""
        self.left = None
        self.right = None
        self.is_leaf = True


class DASTree:
    """
    Decision-aware segmentation tree.

    只负责 segmentation：
    - build():    根据 DR gain 构建 full tree
    - prune_to_M(M): 将 tree 剪枝到 M 个叶子，并给每个 leaf 赋 segment_id
    - assign(X):  对任意 X 返回 segment_id
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        D: np.ndarray,
        gamma: np.ndarray,
        candidate_thresholds,
        min_leaf_size: int,
        max_depth: int,
        epsilon: float = 0.0,
    ):
        """
        Parameters
        ----------
        x : (N, d) features
        y : (N,) outcomes
        D : (N,) treatments (0/1)
        gamma : (N, 2) DR score matrix (Γ_{i0}, Γ_{i1})
        candidate_thresholds : list/tuple of length d,
            each H_j is an iterable of candidate split thresholds for feature j.
        min_leaf_size : int
            每个 leaf 内每个 treatment 至少需要的样本数（StatisticallyAdmissible）
        max_depth : int
            最大树深度 q_max
        epsilon : float
            split gain 阈值，小于等于 epsilon 就不再 split（Algorithm 1 的 ε）
        """
        self.x = x
        self.y = y
        self.D = D
        self.gamma = gamma
        self.H = candidate_thresholds
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth
        self.epsilon = epsilon

        self.root: DASTNode | None = None
        self.leaf_nodes: list[DASTNode] = []

        # 用于 prune 时的浮点比较
        self.tolerance_pruning = 1e-8

    # ======================================================================
    # Public API
    # ======================================================================

    def build(self):
        """Build full tree (Algorithm 1)."""
        all_indices = np.arange(self.x.shape[0])
        self.root = self._grow_node(all_indices, depth=0)

    def prune_to_M(self, M: int):
        """
        Post-prune tree to have exactly M leaves (Algorithm 2,但不做参数估计).

        After this call:
        - self.leaf_nodes: list of leaf nodes
        - each leaf node has .segment_id ∈ {0, …, M-1}
        """
        if self.root is None:
            raise RuntimeError("Call build() before prune_to_M().")

        current_leaves = self._get_leaf_nodes()
        # if len(current_leaves) < M:
        #     raise ValueError(
        #         f"Current leaf count ({len(current_leaves)}) < target M ({M})."
        #     )

        while len(current_leaves) > M:
            # 所有「左右子节点都是叶子」的内部节点
            prunable_nodes = self._get_internal_nodes_with_leaf_children()
            if not prunable_nodes:
                break

            # 对每个候选节点计算 ΔV_L = (V_L + V_R) - V_C
            best_node = None
            best_delta = np.inf
            best_variance_increase = np.inf

            for node in prunable_nodes:
                # 使用缓存的 node.value 提高效率
                V_L = node.left.value if node.left.value is not None else self._compute_node_value(node.left.indices)
                V_R = node.right.value if node.right.value is not None else self._compute_node_value(node.right.indices)
                V_C = node.value if node.value is not None else self._compute_node_value(node.indices)
                delta = (V_L + V_R) - V_C  # Algorithm 2, line 10

                # tie-breaking: variance increase 较小者优先
                if delta < best_delta - self.tolerance_pruning:
                    best_delta = delta
                    best_node = node
                    best_variance_increase = self._compute_variance_after_pruning(node)
                elif abs(delta - best_delta) <= self.tolerance_pruning:
                    var_inc = self._compute_variance_after_pruning(node)
                    if var_inc < best_variance_increase:
                        best_variance_increase = var_inc
                        best_node = node

            if best_node is None:
                break

            # 真正执行 prune
            best_node.prune()
            current_leaves = self._get_leaf_nodes()

        self.leaf_nodes = current_leaves

        # 重新编号 segment_id
        for seg_id, node in enumerate(self.leaf_nodes):
            node.segment_id = seg_id

    def assign(self, X: np.ndarray) -> np.ndarray:
        """
        给任意一批样本 X 分 segment_id。

        要求：已经 build() 且 prune_to_M()，leaf node 上有 segment_id。
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

    # ======================================================================
    # Core growing logic (Algorithm 1)
    # ======================================================================

    def _grow_node(self, indices: np.ndarray, depth: int) -> DASTNode:
        node = DASTNode(indices=indices, depth=depth)
        node.value = self._compute_node_value(indices)

        # Stop if max depth reached
        if depth == self.max_depth:
            self.leaf_nodes.append(node)
            return node

        # Evaluate all candidate splits
        gain_splits, var_splits = self._evaluate_all_splits(node, indices)

        if not gain_splits:
            # 无可行 split（StatisticallyAdmissible 不满足）
            self.leaf_nodes.append(node)
            return node

        # 找到最大的 gain，用于和 epsilon 比较
        max_gain = max(g for (g, _, _, _, _) in gain_splits)
        if max_gain <= self.epsilon:
            # Algorithm 1: if bestGain ≤ ε, TerminateNode
            self.leaf_nodes.append(node)
            return node

        final_split, _ = self._select_best_split(gain_splits, var_splits)
        if final_split is None:
            self.leaf_nodes.append(node)
            return node

        # Execute split
        node.is_leaf = False
        feature, threshold, left_idx, right_idx = final_split
        node.split_feature = feature
        node.split_threshold = threshold
        node.left = self._grow_node(left_idx, depth + 1)
        node.right = self._grow_node(right_idx, depth + 1)

        return node

    def _compute_node_value(self, indices: np.ndarray) -> float:
        """
        ComputeNodeValue(L) — Algorithm 1, lines 25–31.

        For a leaf L:
          - 估计 τ̂_L via difference-in-means
          - a = 1{τ̂_L ≥ 0}
          - v̂_i = y_i if D_i == a else Γ_{i,a}
          - V̂_L = ∑_{i∈L} v̂_i
        """
        if len(indices) == 0:
            return 0.0

        y_L = self.y[indices]
        D_L = self.D[indices]

        n1 = np.sum(D_L == 1)
        n0 = np.sum(D_L == 0)
        if n1 == 0 or n0 == 0:
            # 极端情况：缺某个 treatment，τ̂_L 设为 0
            # 注：这通常不应该发生（有 StatisticallyAdmissible 检查）
            import warnings
            warnings.warn(
                f"Leaf has only one treatment type: n1={n1}, n0={n0}. "
                f"Setting tau_hat=0."
            )
            tau_hat = 0.0
        else:
            y1 = y_L[D_L == 1].mean()
            y0 = y_L[D_L == 0].mean()
            tau_hat = y1 - y0

        a = 1 if tau_hat >= 0.0 else 0

        gamma_L_a = self.gamma[indices, a]
        v_hat = np.where(D_L == a, y_L, gamma_L_a)
        return float(v_hat.sum())

    def _evaluate_all_splits(self, node: DASTNode, indices: np.ndarray):
        """
        Evaluate candidate splits at this node.

        Returns
        -------
        gain_splits: list of (gain, feature, threshold, left_idx, right_idx)
        var_splits:  list of (var_reduction, feature, threshold, left_idx, right_idx)
        """
        gain_splits = []
        var_splits = []

        N, d = self.x.shape

        for j in range(d):
            for t in self.H[j]:
                left_idx = indices[self.x[indices, j] <= t]
                right_idx = indices[self.x[indices, j] > t]

                # StatisticallyAdmissible: leaf 内每个 treatment 至少 min_leaf_size
                if not (self._check_leaf_constraints(left_idx) and
                        self._check_leaf_constraints(right_idx)):
                    continue

                left_val = self._compute_node_value(left_idx)
                right_val = self._compute_node_value(right_idx)
                gain = left_val + right_val - node.value  # Algorithm 1 line 15

                gain_splits.append((gain, j, t, left_idx, right_idx))

                # optional: variance-based tie-breaking
                var_red = self._compute_variance_reduction(indices, left_idx, right_idx)
                var_splits.append((var_red, j, t, left_idx, right_idx))

        return gain_splits, var_splits

    def _select_best_split(self, gain_splits, var_splits):
        """
        在所有 candidate split 中选一个：
        - 先按 gain 最大
        - 若有多条 tie，则按 variance reduction 最大
        """
        if not gain_splits:
            return None, None

        # 先找到最大 gain
        max_gain = max(g for (g, _, _, _, _) in gain_splits)
        # 所有与 max_gain 相差在容差内的 split 都视作 tie
        tied_indices = [
            idx for idx, (g, _, _, _, _) in enumerate(gain_splits)
            if abs(g - max_gain) <= 1e-8
        ]

        if len(tied_indices) == 1:
            _, feature, threshold, left_idx, right_idx = gain_splits[tied_indices[0]]
            return (feature, threshold, left_idx, right_idx), "gain"

        # tie：用 variance reduction 作为第二标准
        best_var = -np.inf
        best_split = None
        for idx in tied_indices:
            var_red, feature, threshold, left_idx, right_idx = var_splits[idx]
            if var_red > best_var:
                best_var = var_red
                best_split = (feature, threshold, left_idx, right_idx)

        if best_split is None:
            return None, None
        return best_split, "gain+variance"

    # ======================================================================
    # Variance utilities (for tie-breaking, pruning)
    # ======================================================================
    def _compute_variance_after_pruning(self, node):
        """
        计算：如果把这个 node 的左右子节点合并（即 prune 掉 split），
        within-cluster 方差会增加多少。

        用在 prune_to_M 里做 tie-breaking：
        - pruning gain（delta）一样时，
        - 选 variance 增加最小的那个 node 来 prune。
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

        # variance increase from pruning
        return merged_var - weighted_before


    def _compute_variance_reduction(self, parent_idx, left_idx, right_idx) -> float:
        """Parent variance minus weighted children variance."""
        parent_var = self._compute_covariate_variance(parent_idx)
        left_var = self._compute_covariate_variance(left_idx)
        right_var = self._compute_covariate_variance(right_idx)

        n_parent = len(parent_idx)
        if n_parent == 0:
            return 0.0

        w_left = len(left_idx) / n_parent
        w_right = len(right_idx) / n_parent
        weighted_child_var = w_left * left_var + w_right * right_var

        return parent_var - weighted_child_var

    def _compute_covariate_variance(self, indices) -> float:
        if len(indices) < 2:
            return 0.0
        return np.var(self.x[indices], axis=0, ddof=1).sum()

    def _check_leaf_constraints(self, indices) -> bool:
        """StatisticallyAdmissible: 每个 treatment 至少 min_leaf_size。"""
        if len(indices) == 0:
            return False
        D_sub = self.D[indices]
        n1 = np.sum(D_sub == 1)
        n0 = np.sum(D_sub == 0)
        return (n1 >= self.min_leaf_size) and (n0 >= self.min_leaf_size)

    # ======================================================================
    # Tree traversal utilities
    # ======================================================================

    def _get_leaf_nodes(self):
        return self._gather_nodes(self.root, lambda n: n.is_leaf)

    def _get_internal_nodes_with_leaf_children(self):
        def cond(n: DASTNode):
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
