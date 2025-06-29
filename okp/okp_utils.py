import numpy as np


def solve_offline_optimal(items, capacity):
    """
    Solve the offline optimal solution using dynamic programming
    """
    n = len(items)
    # dp[i][j] represents the maximum value achievable with first i items and capacity j
    dp = np.zeros((n + 1, capacity + 1))

    for i in range(1, n + 1):
        weight, value = items[i-1]
        for j in range(capacity + 1):
            if weight <= j:
                dp[i][j] = max(dp[i-1][j], dp[i-1][j-weight] + value)
            else:
                dp[i][j] = dp[i-1][j]

    # Backtrack to find selected items
    selected_items = []
    j = capacity
    for i in range(n, 0, -1):
        if dp[i][j] != dp[i-1][j]:
            selected_items.append(items[i-1])
            j -= items[i-1][0]

    return selected_items

# 测试数据集
datasets = {}

# 小规模测试集 (20个物品)
datasets['okp20'] = {
    'okp20_00': {
        'capacity': 100,
        'items': [(20, 40), (30, 60), (40, 80), (10, 20), (25, 50),
                 (15, 30), (35, 70), (45, 90), (5, 10), (50, 100),
                 (12, 24), (28, 56), (32, 64), (18, 36), (22, 44),
                 (38, 76), (42, 84), (8, 16), (33, 66), (27, 54)],
        'optimal': 180  # 最优解的总价值
    }
}

# 中等规模测试集 (50个物品)
datasets['okp50'] = {
    'okp50_00': {
        'capacity': 300,
        'items': [(26, 36), (45, 71), (40, 58), (79, 30), (49, 45), (68, 70), (23, 67), (57, 68), (57, 47), (64, 24), (75, 33), (30, 13), (64, 83), (72, 31), (76, 28), (31, 87), (72, 74), (56, 85), (63, 68), (66, 0), (53, 19), (20, 20), (45, 21), (28, 49), (47, 68), (56, 3), (50, 30), (76, 89), (23, 55), (33, 14), (40, 66), (56, 57), (76, 43), (58, 97), (22, 58), (24, 75), (32, 32), (72, 59), (40, 37), (74, 69), (31, 77), (74, 8), (72, 30), (32, 94), (28, 5), (42, 30), (38, 75), (54, 50), (76, 83), (24, 77)],
        'optimal': 736
    },
    "okp50_01": {
        "capacity": 300,
        "items": [(37, 80), (53, 29), (20, 21), (50, 23), (32, 85), (62, 97), (55, 10), (76, 52), (48, 52), (26, 23), (57, 99), (56, 1), (35, 15), (52, 72), (68, 37), (78, 60), (59, 86), (45, 99), (67, 98), (26, 76), (42, 16), (57, 47), (76, 81), (34, 20), (21, 99), (52, 23), (68, 45), (32, 38), (42, 94), (67, 74), (49, 99), (79, 14), (79, 13), (74, 70), (64, 46), (34, 38), (66, 8), (24, 15), (21, 66), (65, 45), (39, 68), (55, 95), (63, 60), (60, 58), (30, 50), (45, 99), (39, 2), (21, 38), (59, 27), (33, 82)],
        'optimal': 754
    },
    "okp50_02": {
        "capacity": 300,
        "items": [(52, 81), (65, 55), (45, 19), (60, 89), (70, 10), (27, 35), (63, 28), (45, 98), (53, 26), (76, 61), (64, 15), (42, 3), (68, 34), (44, 56), (48, 54), (53, 31), (45, 40), (36, 28), (25, 57), (67, 4), (34, 50), (28, 19), (44, 36), (58, 83), (46, 24), (46, 97), (66, 95), (47, 30), (54, 56), (31, 6), (54, 96), (77, 52), (67, 82), (67, 65), (20, 38), (24, 89), (33, 36), (75, 68), (20, 11), (37, 8), (28, 95), (44, 38), (47, 84), (70, 94), (28, 19), (24, 1), (56, 38), (40, 33), (37, 28), (70, 54)],
        'optimal': 654
    },
    "okp50_03": {
        "capacity": 300,
        "items": [[41, 26], [52, 20], [92, 56], [21, 12], [59, 57], [31, 66], [75, 90], [82, 31], [46, 3], [63, 47], [46, 69], [91, 66], [37, 71], [88, 75], [50, 64], [62, 82], [12, 64], [75, 12], [19, 27], [60, 62], [67, 26], [97, 8], [91, 99], [16, 42], [76, 39], [73, 39], [90, 60], [35, 15], [63, 6], [9, 22], [4, 68], [7, 2], [88, 20], [54, 81], [53, 50], [8, 49], [63, 62], [30, 58], [64, 32], [32, 88], [36, 96], [49, 64], [68, 73], [3, 44], [44, 30], [77, 6], [8, 97], [83, 97], [44, 79], [60, 8]],
        'optimal': 873
    },
    "okp50_04": {
        "capacity": 300,
        "items": [[68, 36], [20, 21], [25, 75], [25, 50], [22, 17], [49, 35], [1, 42], [96, 80], [92, 99], [15, 37], [68, 51], [93, 8], [38, 24], [19, 25], [59, 41], [43, 63], [52, 5], [35, 75], [93, 20], [18, 79], [90, 77], [56, 46], [9, 91], [7, 32], [16, 15], [43, 26], [18, 37], [63, 28], [10, 55], [89, 38], [59, 24], [81, 92], [85, 28], [23, 25], [1, 74], [19, 81], [44, 80], [31, 59], [34, 4], [33, 96], [4, 21], [81, 81], [78, 81], [38, 61], [7, 83], [75, 52], [83, 45], [11, 32], [16, 16], [99, 27]],
        "optimal": 1062
    }
}
