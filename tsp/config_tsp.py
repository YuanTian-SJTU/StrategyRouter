# 策略名称列表
STRATEGIES = [
    'Nearest Neighbor',   # 最近邻策略
    'Random',             # 随机选择下一个城市
    'Farthest Insertion', # 最远插入法（可自定义实现）
]

STRATEGY_DESCRIPTIONS = {
    'Nearest Neighbor': '每次选择距离当前城市最近的未访问城市',
    'Random': '每次随机选择一个未访问城市',
    'Farthest Insertion': '每次插入距离当前路径最远的城市',
} 

STRATEGY_KEYWORDS = {
    'Nearest Neighbor': ['near', 'close'],
    'Random': ['random'],
    'Farthest Insertion': ['far']
}