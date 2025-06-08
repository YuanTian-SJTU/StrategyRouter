# FSSP策略名称列表
STRATEGIES = [
    'Shortest Processing Time',             # Shortest Processing Time - 最短加工时间优先
    'Longest Processing Time',              # Longest Processing Time - 最长加工时间优先
    'Nawaz-Enscore-Ham',                    # Nawaz-Enscore-Ham算法变体
    'Johnson',                              # Johnson规则变体
    'Critical Path',                        # 关键路径法
    'Random',                               # 随机排序
    'First In First Out',                   # First In First Out - 先到先服务
    'Earliest Due Date'                     # Earliest Due Date - 最早交期优先
]

STRATEGY_DESCRIPTIONS = {
    'Shortest Processing Time': '优先调度总加工时间最短的作业',
    'Longest Processing Time': '优先调度总加工时间最长的作业',
    'Nawaz-Enscore-Ham': 'Nawaz-Enscore-Ham启发式算法的变体',
    'Johnson': 'Johnson规则用于两机器流水车间的扩展',
    'Critical Path': '基于关键路径分析的调度策略',
    'Random': '随机选择作业顺序',
    'First In First Out': '按作业到达顺序进行调度',
    'EDD': '按最早交期优先进行调度'
}

STRATEGY_KEYWORDS = {
    'Shortest Processing Time': ['short', 'min', 'small', 'fast'],
    'Longest Processing Time': ['long', 'max', 'large', 'slow'],
    'Nawaz-Enscore-Ham': ['neh', 'insertion', 'Nawaz', 'Enscore', 'Ham'],
    'Johnson': ['johnson', 'pair'],
    'Critical Path': ['critical', 'path', 'bottleneck'],
    'Random': ['random', 'shuffle'],
    'First In First Out': ['fifo', 'first', 'order'],
    'Earliest Due Date': ['due', 'deadline', 'early']
}