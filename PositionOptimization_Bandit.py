import numpy as np
from typing import List, Dict, Any

# 生成来自剩余元素的候选子集
def generate_candidates(remaining_indices: List[int], values: List[int], max_sum: int,
                        pool_size: int = 10, rng: np.random.Generator = None) -> List[Dict[str, Any]]:
    # 若没有给随机数生成器，则创建一个
    if rng is None:
        rng = np.random.default_rng()

    # 三种简单策略，交替产生候选
    def greedy_desc():
        order = sorted(remaining_indices, key=lambda i: values[i], reverse=True)
        s = 0
        subset = []
        for idx in order:
            # 如果当前元素放入后不超 max_sum，则加入
            if s + values[idx] <= max_sum:
                subset.append(idx)
                s += values[idx]
                if s == max_sum:
                    return subset
        return None

    def greedy_asc():
        order = sorted(remaining_indices, key=lambda i: values[i])
        s = 0
        subset = []
        for idx in order:
            if s + values[idx] <= max_sum:
                subset.append(idx)
                s += values[idx]
                if s == max_sum:
                    return subset
        return None

    def random_fill(trials: int = 20):
        for _ in range(trials):
            order = remaining_indices.copy()
            rng.shuffle(order)
            s = 0
            subset = []
            for idx in order:
                if s + values[idx] <= max_sum:
                    subset.append(idx)
                    s += values[idx]
                    if s == max_sum:
                        return subset
        return None

    # 存储已生成的候选子集（避免重复）
    candidates = []
    seen_masks = set()

    for func in (greedy_desc, greedy_asc, random_fill):
        subset = func()
        if subset is not None:
            # mask 的第 i 位为 1 表示包含元素 i
            mask = 0
            for i in subset:
                mask |= (1 << i)
            if mask not in seen_masks:
                candidates.append({'indices': subset, 'mask': mask, 'sum': sum(values[i] for i in subset)})
                seen_masks.add(mask)
        # 若达到 pool_size 上限，则停止生成
        if len(candidates) >= pool_size:
            break

    return candidates


# Bandit（UCB1）实现
class UCBBandit:
    def __init__(self, pool_size: int = 10, seed: int or None = None):
        self.pool_size = pool_size
        self.arms: List[Dict[str, Any]] = []   # 每个臂包含：indices、mask、sum
        self.Q: List[float] = []                # 累计奖励的均值估计
        self.N: List[int] = []                   # 该臂的试验次数
        self.total_pulls: int = 0
        self.rng = np.random.default_rng(seed)

    def add_candidates(self, candidates: List[Dict[str, Any]]):
        for c in candidates:
            self.arms.append(c)
            self.Q.append(0.0)
            self.N.append(0)

    def select_arm(self, feasible_indices: List[int]) -> int or None:
        if not feasible_indices:
            return None
        t = max(1, self.total_pulls)
        best_arm = None
        best_score = -float('inf')
        # 未试验过的臂优先探索
        for a in feasible_indices:
            if self.N[a] == 0:
                score = 1.0  # 促使初期探索
            else:
                avg = self.Q[a] / self.N[a]
                score = avg + np.sqrt(2.0 * np.log(t) / self.N[a])
            # 选择 UCB 分数最高的臂
            if score > best_score:
                best_score = score
                best_arm = a
        return best_arm

    def update(self, arm_idx: int, reward: float):
        self.N[arm_idx] += 1
        n = self.N[arm_idx]
        self.Q[arm_idx] += (reward - self.Q[arm_idx]) / n
        self.total_pulls += 1

    def prune_infeasible(self, remaining_mask: int):
        new_arms = []
        new_Q = []
        new_N = []
        for a, arm in enumerate(self.arms):
            if (arm['mask'] & (~remaining_mask)) != 0:
                # 与剩余元素冲突，淘汰该臂
                continue
            new_arms.append(arm)
            new_Q.append(self.Q[a])
            new_N.append(self.N[a])
        self.arms = new_arms
        self.Q = new_Q
        self.N = new_N


# 主入口：带 bandit 的近似分区
def bandit_partition_approx(values, max_sum: int,
                            pool_size: int = 20,
                            seed: int or None = None,
                            max_rounds: int = 1000):
    """
    values: 一维整数序列，长度可变
    max_sum: 每个子集的目标和
    返回：partition，为一个 list[list[int]]，内部是原始索引（0-based）
    """
    n = len(values)
    # remaining_mask 表示哪些元素还未被选中
    remaining_mask = (1 << n) - 1
    # 存储所有构造出的子集
    partition: List[List[int]] = []

    rng = np.random.default_rng(seed)
    bandit = UCBBandit(pool_size=pool_size, seed=seed)

    rounds = 0
    while remaining_mask != 0 and rounds < max_rounds:
        rounds += 1
        # 获取当前仍“未使用”的元素索引
        remaining_indices = [i for i in range(n) if (remaining_mask & (1 << i)) != 0]

        if not remaining_indices:
            break

        # 生成候选子集
        candidates = generate_candidates(remaining_indices, values, max_sum,
                                       pool_size=pool_size, rng=rng)
        if not candidates:
            break

        bandit.add_candidates(candidates)

        # 找出所有 feasible 的 arms
        feasible = [idx for idx, arm in enumerate(bandit.arms)
                    if (arm['mask'] & (~remaining_mask)) == 0]

        if not feasible:
            break

        arm = bandit.select_arm(feasible)
        if arm is None:
            break

        chosen = bandit.arms[arm]
        partition.append(chosen['indices'])

        # 更新剩余集合
        remaining_mask &= ~chosen['mask']

        # 剪枝：移除与已使用元素冲突的臂
        bandit.prune_infeasible(remaining_mask)

        # 若没有剩余元素，则结束
        if remaining_mask == 0:
            break

    # 将剩余未使用的元素作为单元素子集附加
    leftovers = [i for i in range(n) if (remaining_mask & (1 << i)) != 0]
    for i in leftovers:
        partition.append([i])

    flat_list = [item for sublist in partition for item in sublist]
    return flat_list
