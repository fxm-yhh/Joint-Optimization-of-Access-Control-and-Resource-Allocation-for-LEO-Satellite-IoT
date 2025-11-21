import numpy as np

# 时频图位置优化的一种方法：动态规划
def dp_partition(slotblock_num, max_sum=10):
    n = len(slotblock_num)
    dp = [[False] * (max_sum + 1) for _ in range(n + 1)]
    dp[0][0] = True  # 初始化：和为0的子集始终存在

    # 提取数值用于动态规划计算
    values = slotblock_num['value']

    # 动态规划填表
    for i in range(1, n + 1):
        for j in range(max_sum + 1):
            dp[i][j] = dp[i - 1][j]  # 不选第 i 个数字
            if j >= values[i - 1]:  # 尝试选第 i 个数字
                dp[i][j] |= dp[i - 1][j - values[i - 1]]

    # 回溯找子集
    subsets = []
    current_position_subset = []
    used = [False] * n
    for _ in range(n):  # 循环直到所有数字都被分配
        current_subset = []
        current_sum = max_sum
        # 回溯找到符合条件的子集
        for i in range(n, 0, -1):
            if not used[i - 1] and current_sum >= values[i - 1] and dp[i - 1][current_sum - values[i - 1]]:
                # 添加整个结构化元素（包含值和位置信息）
                current_subset.append(slotblock_num[i - 1])
                current_position_subset.append(slotblock_num[i - 1]['position'])
                current_sum -= values[i - 1]
                used[i - 1] = True

        if not current_subset:  # 如果无法找到新的子集，退出
            break
        subsets.append(current_subset)

    # 把没有加入子集的重新加入
    for i in range(n):
        if not used[i]:
            subsets.append([slotblock_num[i]])
            current_position_subset.append(slotblock_num[i]['position'])

    return len(subsets), subsets, current_position_subset

def zip_data(data):
    positions = np.arange(len(data))
    # 创建结构化数组
    structured_array = np.array(
        list(zip(data, positions)),
        dtype=[('value', 'int'), ('position', 'int')])
    return structured_array

def dp_agent(slot_block, max_sum):
    value_position = zip_data(slot_block)
    value_position = np.sort(value_position, order='value')
    _, _, position_infor = dp_partition(value_position, max_sum=max_sum)
    return position_infor